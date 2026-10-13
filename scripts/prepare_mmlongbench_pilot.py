"""Create a reproducible 40-question DEV pilot for local GPU diagnosis.

This resource-bounded, balanced subset is not a representative test benchmark.
Whole PDFs are retained; the page limit selects documents, never truncates them.
"""
import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.dataset_loader import audit_documents, load_eval_dataset
from src.eval.provenance import atomic_json, file_sha256

QUOTAS = {"unanswerable": 8, "text": 6, "table": 6, "chart": 6,
          "figure": 6, "mixed": 4, "layout": 4}


def select_pilot(items, documents, seed=42, max_document_pages=80, quotas=None):
    quotas = QUOTAS if quotas is None else quotas
    if max_document_pages < 1 or any(n < 1 for n in quotas.values()):
        raise ValueError("Page limit and category quotas must be positive")
    candidates = [i for i in sorted(items, key=lambda i: i["id"])
                  if documents[i["pdf_path"]]["pages"] <= max_document_pages
                  and i.get("evidence_annotation_status", "complete") in {"complete", "not_applicable"}]
    random.Random(seed).shuffle(candidates)
    tie_order = {i["id"]: index for index, i in enumerate(candidates)}
    selected, doc_counts, type_counts = [], Counter(), Counter()
    # Draw the less abundant categories first. Prefer covering document types,
    # then reuse selected PDFs (up to four questions) to limit indexing cost.
    for category in sorted(quotas, key=lambda c: (sum(i.get("category") == c for i in candidates), c)):
        pool = [i for i in candidates if i.get("category") == category]
        if len(pool) < quotas[category]:
            raise ValueError(f"Not enough eligible {category} questions: {len(pool)} < {quotas[category]}")
        cross_target = min(2, quotas[category], sum(len(i["gold_pages"]) > 1 for i in pool))
        cross_selected = 0
        for _ in range(quotas[category]):
            preferred = [i for i in pool if (len(i["gold_pages"]) > 1) == (cross_selected < cross_target)]
            options = preferred or pool
            item = min(options, key=lambda i: (
                doc_counts[i["document_sha256"]] >= 4,
                type_counts[i.get("doc_type", "unknown")],
                0 if doc_counts[i["document_sha256"]] else 1,
                documents[i["pdf_path"]]["pages"], tie_order[i["id"]]))
            selected.append(item)
            pool.remove(item)
            doc_counts[item["document_sha256"]] += 1
            type_counts[item.get("doc_type", "unknown")] += 1
            cross_selected += len(item["gold_pages"]) > 1
    # Group adjacent questions by PDF for index-cache reuse.
    return sorted(selected, key=lambda i: (i["doc_id"], i["id"]))


def prepare_pilot(dataset_path, output_path, seed=42, max_document_pages=80):
    dataset_path, output_path = Path(dataset_path).resolve(), Path(output_path).resolve()
    if dataset_path == output_path:
        raise ValueError("Pilot output must not overwrite its source dataset")
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("split") != "dev":
        raise ValueError("Pilot selection requires a DEV dataset; never use test or full")
    items = load_eval_dataset(dataset_path)
    documents = audit_documents(items)
    selected = select_pilot(items, documents, seed, max_document_pages)
    selected_docs = {i["pdf_path"] for i in selected}
    report = {
        "source_dataset": os.path.relpath(dataset_path, output_path.parent).replace("\\", "/"),
        "source_dataset_sha256": file_sha256(dataset_path), "seed": seed,
        "max_document_pages": max_document_pages, "quotas": QUOTAS,
        "questions": len(selected), "documents": len(selected_docs),
        "total_pages_to_index": sum(documents[p]["pages"] for p in selected_docs),
        "answerable": sum(i["answerable"] for i in selected),
        "unanswerable": sum(not i["answerable"] for i in selected),
        "cross_page": sum(len(i["gold_pages"]) > 1 for i in selected),
        "categories": dict(Counter(i["category"] for i in selected)),
        "document_types": dict(Counter(i.get("doc_type") for i in selected)),
        "selection_note": "DEV-only balanced diagnostic sample; excludes PDFs >80 pages by default and unusable evidence labels; not a test score.",
    }
    result = {key: payload[key] for key in ("name", "schema_version", "source", "page_numbering", "evaluation_note") if key in payload}
    result.update({"split": "pilot_from_dev", "selection": report,
                   "items": [{**i, "pdf_path": os.path.relpath(i["pdf_path"], output_path.parent).replace("\\", "/")} for i in selected]})
    atomic_json(output_path, result)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/eval/mmlongbench/dev.json")
    parser.add_argument("--output", default="data/eval/mmlongbench/pilot40.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-document-pages", type=int, default=80)
    args = parser.parse_args()
    print(json.dumps(prepare_pilot(args.dataset, args.output, args.seed, args.max_document_pages), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
