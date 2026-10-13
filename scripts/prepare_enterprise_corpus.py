"""Copy local enterprise PDFs, preserve provenance, and extract page-level evidence.

This is corpus preparation, not automatic gold-label generation. PDF text is data.
"""
import argparse
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.provenance import atomic_json, file_sha256


def prepare(source_dir, output_dir, specs_path):
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    specs = json.loads(Path(specs_path).read_text(encoding="utf-8"))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Use an empty output directory; existing annotations must be preserved")
    actual = {p.name for p in source_dir.glob("*.pdf")}
    expected = {s["filename"] for s in specs["documents"]}
    if actual != expected:
        raise ValueError(f"Source PDF names differ: missing={expected-actual}; extra={actual-expected}")
    group_splits, hash_splits, documents, ids, inputs = {}, {}, [], set(), []
    # Preflight the entire split before copying anything.
    for spec in specs["documents"]:
        ident = spec["doc_id"]
        if not re.fullmatch(r"[a-z][a-z0-9_]*", ident) or ident in ids:
            raise ValueError(f"Invalid/duplicate document id: {ident}")
        ids.add(ident)
        if Path(spec["filename"]).name != spec["filename"] or spec["split"] not in {"dev", "test", "supplementary"}:
            raise ValueError("Invalid source filename or split")
        source = source_dir / spec["filename"]
        digest = file_sha256(source)
        for groups, key in ((group_splits, spec["company_group"]), (hash_splits, digest)):
            if key in groups and groups[key] != spec["split"]:
                raise ValueError(f"Document/company leakage: {source.name}")
            groups[key] = spec["split"]
        with pymupdf.open(source) as doc:
            if doc.needs_pass or not len(doc):
                raise ValueError(f"PDF is empty or needs a password: {source.name}")
        inputs.append((spec, source, digest))
    for spec, source, digest in inputs:
        target = output_dir / "pdfs" / f"{spec['doc_id']}.pdf"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if file_sha256(target) != digest:
            raise ValueError(f"Copy hash mismatch: {source}")
        page_dir = output_dir / "corpus" / spec["doc_id"]
        page_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        with pymupdf.open(target) as doc:
            if doc.needs_pass:
                raise ValueError(f"Encrypted PDF needs a password: {source.name}")
            for index, page in enumerate(doc):
                text = page.get_text("text", sort=True)
                blocks = [{"bbox": [round(v, 2) for v in b[:4]], "text": b[4]}
                          for b in page.get_text("blocks", sort=True) if b[6] == 0]
                compact = "".join(text.split())
                # Signals are only inspection aids, never table/chart ground truth.
                numbers = sum(c.isdigit() for c in compact)
                row = {"doc_id": spec["doc_id"], "page": index + 1,
                       "width_pt": page.rect.width, "height_pt": page.rect.height,
                       "text": text, "blocks": blocks, "characters": len(compact),
                       "image_objects": len(page.get_images()),
                       "digit_fraction": round(numbers / max(1, len(compact)), 3),
                       "needs_visual_check": len(compact) < 80 or "\ufffd" in text}
                rows.append(row)
            count = len(doc)
        with (page_dir / "pages.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        (page_dir / "pages.txt").write_text("\n\n".join(
            f"=== PDF PHYSICAL PAGE {r['page']} ===\n{r['text']}" for r in rows), encoding="utf-8")
        documents.append({**spec, "source_path": str(source), "source_url": None,
                          "source_verification": "user_supplied_not_independently_authenticated",
                          "redistribution_permission": "not_verified",
                          "pdf_path": target.relative_to(output_dir).as_posix(),
                          "sha256": digest, "pages": count, "bytes": target.stat().st_size,
                          "text_characters": sum(r["characters"] for r in rows),
                          "low_text_pages": [r["page"] for r in rows if r["characters"] < 80],
                          "replacement_character_pages": [r["page"] for r in rows if "\ufffd" in r["text"]],
                          "pages_with_image_objects": sum(bool(r["image_objects"]) for r in rows)})
        print(json.dumps({"doc_id": spec["doc_id"], "pages": count,
                          "low_text_pages": documents[-1]["low_text_pages"]}, ensure_ascii=False), flush=True)
    manifest = {"schema_version": 1, "name": "enterprise_cn_pilot_v1",
                "page_numbering": "1-based physical PDF pages, including covers",
                "split_policy": specs["split_policy"], "documents": documents,
                "summary": {"pdfs": len(documents), "pages": sum(d["pages"] for d in documents),
                            "distinct_hashes": len(hash_splits),
                            "split_documents": dict(Counter(d["split"] for d in documents)),
                            "split_pages": {s: sum(d["pages"] for d in documents if d["split"] == s)
                                            for s in sorted({d["split"] for d in documents})}}}
    atomic_json(output_dir / "manifest.json", manifest)
    return manifest["summary"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", default="data/enterprise_cn")
    parser.add_argument("--specs", default="docs/enterprise_cn_sources.json")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_dir, args.output_dir, args.specs), ensure_ascii=False, indent=2))
