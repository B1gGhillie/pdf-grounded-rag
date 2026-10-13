"""Export MMLongBench-Doc into portable, document-disjoint project data.

All questions are retained. Missing/invalid evidence annotations are explicitly
masked for page metrics, never guessed or converted into unanswerable questions.
"""
import argparse
import ast
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.dataset_loader import audit_documents, split_by_document, validate_items
from src.eval.provenance import atomic_json, file_sha256


def _as_list(value):
    if value is None or value == "":
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            # Official evidence_sources uses Python list strings, not JSON.
            value = ast.literal_eval(value)
    if not isinstance(value, (list, tuple)):
        raise ValueError("Expected a list annotation")
    return list(value)


def _as_pages(value):
    if type(value) is int:
        value = [value]
    pages = _as_list(value)
    if any(type(page) is not int or page < 1 for page in pages):
        raise ValueError("Evidence pages must be positive 1-based integers")
    return sorted(set(pages))


def normalize_item(raw, idx, pdf_root, page_count=None):
    question = raw.get("question") or raw.get("query")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Row {idx}: missing question")
    answer = raw.get("answer", "")
    if not isinstance(answer, str):
        answer = json.dumps(answer, ensure_ascii=False)
    upstream_unanswerable = (answer.strip().lower() == "not answerable"
                            or raw.get("answer_format") == "None")
    answerable = raw.get("answerable", not upstream_unanswerable)
    if type(answerable) is not bool or (answerable and upstream_unanswerable):
        raise ValueError(f"Row {idx}: inconsistent answerability labels")
    doc_id = raw.get("doc_id") or raw.get("document_id")
    if not doc_id:
        raise ValueError(f"Row {idx}: missing doc_id")
    pdf_name = str(doc_id)
    if not pdf_name.lower().endswith(".pdf"):
        pdf_name += ".pdf"
    pdf_root = Path(pdf_root).resolve()
    if Path(pdf_name).name != pdf_name or "\\" in pdf_name:
        raise ValueError(f"Row {idx}: doc_id must be a PDF filename")
    pdf_path = pdf_root / pdf_name
    upstream_pages = raw.get("gold_pages", raw.get("evidence_pages", []))
    issue = None
    if not answerable:
        pages, status = [], "not_applicable"
    else:
        try:
            pages = _as_pages(upstream_pages)
            status = "complete" if pages else "missing"
            if not pages:
                issue = "Upstream answerable question has no evidence-page annotation"
            elif page_count is not None and any(p > page_count for p in pages):
                raise ValueError(f"Evidence page exceeds PDF length ({page_count})")
        except (ValueError, SyntaxError, TypeError) as exc:
            pages, status, issue = [], "invalid", str(exc)
    sources = _as_list(raw.get("evidence_sources", []))
    if any(not isinstance(source, str) for source in sources):
        raise ValueError(f"Row {idx}: evidence sources must be strings")
    categories = {"Pure-text (Plain-text)": "text", "Generalized-text (Layout)": "layout",
                  "Table": "table", "Chart": "chart", "Figure": "figure"}
    if not answerable:
        category = "unanswerable"
    elif len(set(sources)) > 1:
        category = "mixed"
    else:
        category = categories.get(sources[0], "unspecified") if sources else "unspecified"
    item = {
        "id": str(raw.get("id") or raw.get("qid") or f"mmlongbench_{idx:05d}"),
        "source_index": idx, "doc_id": str(doc_id), "pdf_path": str(pdf_path),
        "question": question, "answer": answer if answerable else "",
        "answerable": answerable, "gold_pages": pages,
        "evidence_annotation_status": status, "category": category,
        "doc_type": raw.get("doc_type"), "answer_format": raw.get("answer_format"),
        "evidence_sources": sources, "upstream_answer": raw.get("answer"),
        "upstream_evidence_pages": upstream_pages,
        "upstream_evidence_sources": raw.get("evidence_sources"),
    }
    if issue:
        item["evidence_annotation_issue"] = issue
    return item


def load_raw_records(path):
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("items", "data", "examples", "questions"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"Unrecognized JSON structure: {path}")


def _portable(items, directory):
    return [{**item, "pdf_path": os.path.relpath(item["pdf_path"], directory).replace("\\", "/")}
            for item in items]


def summarize(items):
    return {
        "questions": len(items), "documents": len({i["doc_id"] for i in items}),
        "unique_pdf_hashes": len({i["document_sha256"] for i in items}),
        "answerable": sum(i["answerable"] for i in items),
        "unanswerable": sum(not i["answerable"] for i in items),
        "with_usable_evidence": sum(i["evidence_annotation_status"] == "complete" for i in items),
        "cross_page_with_usable_evidence": sum(len(i["gold_pages"]) > 1 for i in items),
        "evidence_status": dict(Counter(i["evidence_annotation_status"] for i in items)),
        "categories": dict(Counter(i["category"] for i in items)),
        "document_types": dict(Counter(i["doc_type"] for i in items)),
    }


def convert_from_json(raw_path, output_path, pdf_root, limit=None, dev_fraction=0.3, seed=42):
    import pymupdf

    records = load_raw_records(raw_path)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        records = records[:limit]
    items, page_counts = [], {}
    for idx, raw in enumerate(records):
        item = normalize_item(raw, idx, pdf_root)
        path = item["pdf_path"]
        if path not in page_counts:
            if not Path(path).is_file():
                raise FileNotFoundError(f"Missing PDF: {path}. Run with --from-hf to download the snapshot.")
            with pymupdf.open(path) as pdf:
                if pdf.needs_pass or len(pdf) < 1:
                    raise ValueError(f"Unreadable/empty PDF: {path}")
                page_counts[path] = len(pdf)
        items.append(normalize_item(raw, idx, pdf_root, page_counts[path]))
    validate_items(items)
    docs = audit_documents(items)
    output_path = Path(output_path).resolve()
    out = output_path.parent
    source_manifest = Path(raw_path).parent / "source_manifest.json"
    source = json.loads(source_manifest.read_text(encoding="utf-8")) if source_manifest.exists() else {}
    if Path(raw_path).name != "samples.hf.json":
        source = {}
    if source and source.get("raw_json_sha256") != file_sha256(raw_path):
        raise ValueError("Raw JSON differs from the downloaded snapshot manifest")
    source = {key: source[key] for key in ("dataset", "revision", "source_url", "license", "annotation_sha256") if key in source}
    if source:
        source["source_manifest_sha256"] = file_sha256(source_manifest)
    source["raw_json_sha256"] = file_sha256(raw_path)
    metadata = {
        "name": "MMLongBench-Doc", "schema_version": 2, "source": source,
        "page_numbering": "1-based physical PDF pages; invalid annotations are masked, not shifted",
        "evaluation_note": "Project EM/token-F1 and refusal/page metrics; not the official MMLongBench scorer.",
    }
    atomic_json(output_path, {**metadata, "split": "full", "items": _portable(items, out)})
    splits = {}
    if len({i["document_sha256"] for i in items}) >= 2:
        splits = split_by_document(items, dev_fraction=dev_fraction, seed=seed)
        for name, rows in splits.items():
            atomic_json(out / f"{name}.json", {**metadata, "split": name,
                "split_origin": "project-created, not an official benchmark partition",
                "seed": seed, "dev_fraction": dev_fraction, "items": _portable(rows, out)})
        dev_docs = {i["pdf_path"] for i in splits["dev"]}
        both_labels = [p for p in dev_docs if {i["answerable"] for i in splits["dev"] if i["pdf_path"] == p} == {True, False}]
        small_doc = min(both_labels or dev_docs, key=lambda p: (page_counts[p], Path(p).name))
        candidates = [i for i in splits["dev"] if i["pdf_path"] == small_doc]
        smoke = sorted(candidates, key=lambda i: (i["answerable"], i["source_index"]))[:1]
        smoke += [i for i in candidates if i["answerable"]][:3]
        smoke = list({i["id"]: i for i in smoke}.values())
        atomic_json(out / "smoke.json", {**metadata, "split": "smoke_from_dev", "items": _portable(smoke, out)})
    duplicates = Counter((i["doc_id"], i["question"]) for i in items)
    hash_groups = {}
    for path, info in docs.items():
        hash_groups.setdefault(info["sha256"], []).append(Path(path).name)
    report = {
        "source": source, "full": summarize(items),
        "splits": {name: summarize(rows) for name, rows in splits.items()},
        "split_seed": seed, "dev_document_fraction": dev_fraction,
        "split_document_overlap": sorted({i["document_sha256"] for i in splits.get("dev", [])}
                                          & {i["document_sha256"] for i in splits.get("test", [])}),
        "total_pdf_pages": sum(d["pages"] for d in docs.values()),
        "max_pdf_pages": max(d["pages"] for d in docs.values()),
        "annotation_issues": [{"id": i["id"], "doc_id": i["doc_id"], "source_index": i["source_index"],
            "raw_evidence_pages": i["upstream_evidence_pages"], "issue": i["evidence_annotation_issue"]}
            for i in items if i.get("evidence_annotation_issue")],
        "duplicate_questions_retained": [{"doc_id": d, "question": q, "count": n} for (d, q), n in duplicates.items() if n > 1],
        "duplicate_pdf_content_groups": [names for names in hash_groups.values() if len(names) > 1],
        "unanswerable_source_evidence_retained_as_metadata": sum(not i["answerable"] and bool(_as_list(i["upstream_evidence_pages"])) for i in items),
        "documents": [{"filename": Path(p).name, **info} for p, info in sorted(docs.items())],
    }
    atomic_json(out / "manifest.json", report)
    return {"output": str(output_path), "full": report["full"], "splits": report["splits"],
            "annotation_issues": len(report["annotation_issues"]), "total_pdf_pages": report["total_pdf_pages"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--from-json", help="Local official JSON/JSONL annotations")
    mode.add_argument("--from-hf", "--download", action="store_true", help="Download pinned annotations and all PDFs")
    parser.add_argument("--raw-dir", default="data/raw/mmlongbench")
    parser.add_argument("--pdf-root", default="data/pdfs/mmlongbench")
    parser.add_argument("--output", default="data/eval/mmlongbench/full.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dev-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    raw_path = args.from_json
    if args.from_hf:
        from scripts.download_mmlongbench import download_snapshot
        raw_path = download_snapshot(args.raw_dir, args.pdf_root)
    stats = convert_from_json(raw_path, args.output, args.pdf_root, args.limit, args.dev_fraction, args.seed)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
