"""Download a pinned official MMLongBench-Doc snapshot, including every PDF.

Files are checked against Hugging Face Git/LFS hashes before being reused.
Install requirements-data.txt to read the original Parquet annotations.
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.provenance import atomic_json, file_sha256

DATASET = "yubo2333/MMLongBench-Doc"
REVISION = "2ff6aa9237fc777b6627dc57a486e9225ac5fb86"
GITHUB_REVISION = "d73f0dc0be7e0a2ff6a403d5fe65fcd96461f384"
BASE = "https://huggingface.co"


def fetch_bytes(url):
    request = Request(url, headers={"User-Agent": "pdf-grounded-rag-dataset-import/1.0"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def valid_download(path, entry):
    path = Path(path)
    if not path.is_file():
        return False
    if entry.get("lfs"):
        # Some upstream size metadata is stale. The content hash is authoritative.
        return file_sha256(path) == entry["lfs"]["oid"]
    if path.stat().st_size != entry["size"]:
        return False
    digest = hashlib.sha1()
    digest.update(f"blob {entry['size']}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == entry["oid"]


def download_file(entry, destination, revision, url=None):
    destination = Path(destination)
    if valid_download(destination, entry):
        return "cached"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    url = url or f"{BASE}/datasets/{DATASET}/resolve/{revision}/{quote(entry['path'], safe='/')}?download=true"
    for attempt in range(3):
        try:
            # A fresh query on retry avoids reusing an inconsistent CDN redirect.
            separator = "&" if "?" in url else "?"
            request_url = url if attempt == 0 else url + f"{separator}retry={time.time_ns()}"
            request = Request(request_url, headers={"User-Agent": "pdf-grounded-rag-dataset-import/1.0",
                                                    "Cache-Control": "no-cache"})
            with urlopen(request, timeout=60) as response, temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream, length=1024 * 1024)
            if not valid_download(temporary, entry):
                raise ValueError(f"Checksum/size mismatch: {entry['path']} (expected {entry['size']}, received {temporary.stat().st_size} bytes)")
            temporary.replace(destination)
            return "downloaded"
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def download_pdf(entry, destination, revision, github_entries):
    """Use the authors' pinned GitHub mirror if HF serves inconsistent bytes.

    A mirror file is validated against its own Git blob hash and its provenance
    is recorded explicitly; it is never declared to match a different HF hash.
    """
    hf_source = {"provider": "huggingface", "revision": revision, "path": entry["path"]}
    if valid_download(destination, entry):
        return hf_source
    mirror = github_entries.get("data/" + entry["path"])
    mirror_entry = {"size": mirror["size"], "oid": mirror["sha"], "path": mirror["path"]} if mirror else None
    mirror_source = {"provider": "github", "revision": GITHUB_REVISION,
                     "path": mirror["path"], "git_blob_sha1": mirror["sha"],
                     "reason": "HF resolve response did not match its advertised size/hash"} if mirror else None
    if mirror_entry and valid_download(destination, mirror_entry):
        return mirror_source
    try:
        download_file(entry, destination, revision)
        return hf_source
    except (OSError, ValueError):
        if not mirror_entry:
            raise
        print(f"Using verified official GitHub mirror: {entry['path']}", flush=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        if valid_download(temporary, mirror_entry):
            temporary.replace(destination)
        else:
            url = f"https://raw.githubusercontent.com/mayubo2333/MMLongBench-Doc/{GITHUB_REVISION}/{quote(mirror['path'], safe='/')}"
            download_file(mirror_entry, destination, GITHUB_REVISION, url=url)
        return mirror_source


def download_snapshot(raw_dir, pdf_root, revision=REVISION, workers=4):
    import pyarrow.parquet as parquet

    raw_dir, pdf_root = Path(raw_dir), Path(pdf_root)
    raw_dir.mkdir(parents=True, exist_ok=True)
    pdf_root.mkdir(parents=True, exist_ok=True)
    tree_url = f"{BASE}/api/datasets/{DATASET}/tree/{revision}?recursive=true&limit=1000"
    entries = json.loads(fetch_bytes(tree_url))
    # This fixed snapshot has fewer than 1000 entries. Refuse truncated future trees.
    if len(entries) >= 1000:
        raise ValueError("Repository tree may be truncated; pagination is required")
    atomic_json(raw_dir / "hf_tree.json", entries)
    github_tree = json.loads(fetch_bytes(f"https://api.github.com/repos/mayubo2333/MMLongBench-Doc/git/trees/{GITHUB_REVISION}?recursive=1"))
    if github_tree.get("truncated"):
        raise ValueError("GitHub mirror tree is truncated")
    atomic_json(raw_dir / "github_tree.json", github_tree)
    github_entries = {e["path"]: e for e in github_tree["tree"] if e["type"] == "blob"}
    annotation = next(e for e in entries if e["path"] == "data/train-00000-of-00001.parquet")
    download_file(annotation, raw_dir / "annotations.parquet", revision)
    records = parquet.read_table(raw_dir / "annotations.parquet").to_pylist()
    atomic_json(raw_dir / "samples.hf.json", records)
    print(f"Pinned annotations: {len(records)} questions", flush=True)

    files = []
    for entry in entries:
        remote = PurePosixPath(entry["path"])
        if entry["type"] == "file" and remote.suffix.lower() == ".pdf":
            if len(remote.parts) != 2 or remote.parts[0] != "documents" or ".." in remote.parts:
                raise ValueError(f"Unexpected PDF path: {remote}")
            files.append((entry, pdf_root / remote.name))
    missing = {r["doc_id"] for r in records} - {p.name for _, p in files}
    if missing:
        raise ValueError(f"Missing official PDF entries: {sorted(missing)}")
    required = sum(e["size"] for e, p in files if not p.exists())
    if shutil.disk_usage(pdf_root).free < required + 100 * 1024 * 1024:
        raise OSError("Insufficient free disk space for the full PDF snapshot")
    print(f"Checking/downloading {len(files)} PDFs ({sum(e['size'] for e, _ in files) / 1024**2:.1f} MiB)", flush=True)
    failures, file_sources = [], {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(download_pdf, entry, path, revision, github_entries): entry["path"] for entry, path in files}
        for completed, future in enumerate(as_completed(pending), 1):
            try:
                file_sources[pending[future]] = future.result()
            except Exception as exc:
                failures.append({"path": pending[future], "error": str(exc)})
                print(f"FAILED {pending[future]}: {type(exc).__name__}", flush=True)
            if completed % 10 == 0 or completed == len(files):
                print(f"PDF progress: {completed}/{len(files)}; failures: {len(failures)}", flush=True)
    if failures:
        atomic_json(raw_dir / "download_failures.json", failures)
        raise RuntimeError(f"{len(failures)} PDF downloads failed; rerun to reuse verified files")

    readme_url = f"https://raw.githubusercontent.com/mayubo2333/MMLongBench-Doc/{GITHUB_REVISION}/README.md"
    (raw_dir / "README.upstream.md").write_bytes(fetch_bytes(readme_url))
    manifest = {
        "dataset": DATASET, "revision": revision, "upstream_split": "train",
        "upstream_split_note": "The upstream split name is train; this is a benchmark, not a prescribed training partition.",
        "source_url": f"{BASE}/datasets/{DATASET}/tree/{revision}",
        "readme_url": readme_url, "license": "CC-BY-NC-4.0",
        "annotation_file": annotation["path"],
        "annotation_sha256": file_sha256(raw_dir / "annotations.parquet"),
        "raw_json_sha256": file_sha256(raw_dir / "samples.hf.json"),
        "questions": len(records), "pdf_count": len(files),
        "pdf_bytes": sum(e["size"] for e, _ in files),
        "documents": [{"filename": p.name, "size": p.stat().st_size, "sha256": file_sha256(p),
                       "hf_reported_size": e["size"], "matches_hf_hash": valid_download(p, e),
                       "source": file_sources[e["path"]]} for e, p in files],
    }
    manifest["pdf_bytes"] = sum(p.stat().st_size for _, p in files)
    atomic_json(raw_dir / "source_manifest.json", manifest)
    (raw_dir / "download_failures.json").unlink(missing_ok=True)
    print(f"Verified complete snapshot: {raw_dir / 'source_manifest.json'}", flush=True)
    return raw_dir / "samples.hf.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/mmlongbench")
    parser.add_argument("--pdf-root", default="data/pdfs/mmlongbench")
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be between 1 and 16")
    download_snapshot(args.raw_dir, args.pdf_root, args.revision, args.workers)


if __name__ == "__main__":
    main()
