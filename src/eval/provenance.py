"""Small reproducibility record; contains no credentials or environment dump."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def collect_provenance(cfg, dataset_path, items):
    packages = {}
    for name in ("torch", "transformers", "accelerate", "bitsandbytes", "pymupdf", "rank-bm25", "numpy", "Pillow", "colpali-engine"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            return b""
    # Hash current source, including untracked implementation files.
    sources = {p.relative_to(root).as_posix(): file_sha256(p)
               for folder in ("src", "scripts", "configs") for p in (root/folder).rglob("*")
               if p.suffix in {".py", ".yaml"}}
    hardware = {"processor": platform.processor()}
    try:
        import torch
        hardware.update({"cuda_version": torch.version.cuda, "cuda_available": torch.cuda.is_available()})
        if torch.cuda.is_available():
            hardware["gpus"] = [{"name": torch.cuda.get_device_name(i), "vram_bytes": torch.cuda.get_device_properties(i).total_memory}
                                for i in range(torch.cuda.device_count())]
    except (ImportError, OSError):
        hardware["cuda_available"] = False
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "config": cfg,
            "config_sha256": json_hash(cfg), "dataset_sha256": file_sha256(dataset_path),
            "document_sha256": sorted({i["document_sha256"] for i in items}),
            "python": platform.python_version(), "platform": platform.platform(),
            "hardware": hardware,
            "packages": packages, "git_commit": git("rev-parse", "HEAD").decode().strip(),
            "git_dirty": bool(git("status", "--porcelain")), "source_sha256": json_hash(sources),
            "seed": cfg.get("evaluation", {}).get("seed", 42), "sampling": "greedy (do_sample=False)",
            "note": "Exact model revision fields should be pinned before scientific comparison."}
