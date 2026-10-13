"""Download pinned official Chinese FinGLM annotations and selected report PDFs."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REVISION = "ff48a9dc5e6c7f9d9a9c6120024dccefd8cbaa80"
MS_REVISION = "0d1da4119ba6f84124c5d71d1275c9502dbdb71c"
MS_BASE = "https://www.modelscope.cn/api/v1/datasets/modelscope/chatglm_llm_fintech_raw_dataset"
SOURCE_FILES = {
    "pre.jsonl": "data/pre-data/answer.json",
    "a.jsonl": "data/A-data/A-list-answer.json",
    "b.jsonl": "data/B-data/B-list-answer.json",
    "c.jsonl": "data/C-data/C-list-answer.json",
    "reports_list.csv": "data/reports/reports_list.csv",
    "README.upstream.md": "README.md",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def download(url, path, expected_sha=None, pdf=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (not expected_sha or sha256(path) == expected_sha):
        return {"path": str(path), "url": url, "bytes": path.stat().st_size, "sha256": sha256(path)}
    temp = path.with_suffix(path.suffix + ".part")
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pdf-grounded-rag-research/1.0"})
            with urllib.request.urlopen(req, timeout=90) as response, temp.open("wb") as out:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    out.write(chunk)
            if pdf and not temp.read_bytes()[:5] == b"%PDF-":
                raise ValueError("Response is not a PDF")
            actual = sha256(temp)
            if expected_sha and actual != expected_sha:
                raise ValueError("Downloaded file SHA256 disagrees with upstream index")
            temp.replace(path)
            print(f"downloaded {path.name} ({path.stat().st_size / 2**20:.1f} MiB)", flush=True)
            return {"path": str(path), "url": url, "bytes": path.stat().st_size, "sha256": actual}
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/finglm_cn"))
    ap.add_argument("--stage", choices=["annotations", "pdfs"], default="annotations")
    args = ap.parse_args()
    root = args.root.resolve()
    if args.stage == "annotations":
        tasks = [(f"https://raw.githubusercontent.com/MetaGLM/FinGLM/{REVISION}/{remote}", root / "raw" / name)
                 for name, remote in SOURCE_FILES.items()]
        tasks.append((MS_BASE + "/repo?" + urllib.parse.urlencode({"Revision": MS_REVISION, "FilePath": "train.csv"}),
                      root / "raw" / "modelscope_train.csv"))
        tasks.append((MS_BASE + "/repo/tree?" + urllib.parse.urlencode({"Revision": MS_REVISION, "Root": "allpdf", "PageSize": 100, "PageNumber": 1}),
                      root / "raw" / "pdf_tree_sample.json"))
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(lambda task: download(*task), tasks))
        report = {"dataset": "MetaGLM/FinGLM", "revision": REVISION, "modelscope_revision": MS_REVISION,
                  "language": "zh", "source": "https://github.com/MetaGLM/FinGLM", "files": results}
        (root / "source_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        plan = json.loads((root / "download_plan.json").read_text(encoding="utf-8"))
        def fetch(item):
            url = item.get("url") or (MS_BASE + "/repo?" + urllib.parse.urlencode({
                "Revision": MS_REVISION, "FilePath": "allpdf/" + item["source_name"]
            }))
            if not url.startswith(("https://www.modelscope.cn/", "https://sail-moe.oss-cn-hangzhou.aliyuncs.com/")):
                raise ValueError("Unrecognized PDF source domain")
            result = download(url, root / "pdfs" / (item["doc_id"] + ".pdf"), item.get("upstream_sha256"), pdf=True)
            return {**item, **result}
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(fetch, plan["documents"]))
        (root / "pdf_manifest.json").write_text(json.dumps({"documents": results}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"PDFs ready: {len(results)}", flush=True)


if __name__ == "__main__":
    main()
