"""Join gold labels and raw predictions for DEV error diagnosis without inference.

Raw EM/F1 are lexical diagnostics, not human-verified correctness or official
MMLongBench scores. Partial runs can be inspected but are labeled incomplete.
"""
import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.dataset_loader import load_eval_dataset
from src.eval.metrics import exact_match, token_f1, score_prediction
from src.eval.provenance import atomic_json, file_sha256
from src.grounding.citation import extract_cited_pages, is_unanswerable, page_number_from_hit


def diagnose(run, items):
    gold = {i["id"]: i for i in items}
    predictions = run.get("predictions", [])
    ids = [p["id"] for p in predictions]
    if len(gold) != len(items) or len(set(ids)) != len(ids):
        raise ValueError("Duplicate gold or prediction IDs")
    if set(ids) - set(gold):
        raise ValueError("Prediction IDs do not belong to the supplied dataset")
    if run.get("status") == "complete" and set(ids) != set(gold):
        raise ValueError("Completed run must contain every supplied dataset ID")
    rows = []
    for prediction in predictions:
        item = gold[prediction["id"]]
        raw = prediction.get("raw_answer")
        generated = bool(raw and raw.strip())
        abstained = generated and is_unanswerable(raw)
        pages = [page_number_from_hit(p) for p in prediction.get("retrieved_pages", [])]
        gold_pages = item["gold_pages"]
        recall = len(set(pages) & set(gold_pages)) / len(set(gold_pages)) if gold_pages else None
        raw_em = exact_match(raw, item["answer"]) if item["answerable"] and generated and not abstained else None
        grounding = prediction.get("grounding") or {}
        metric = score_prediction(prediction, item)
        if recall is not None and recall < 1:
            triage = "retrieval_missing_gold_pages"
        elif not generated:
            triage = "no_candidate_generated"
        elif not item["answerable"]:
            triage = "unanswerable_model_abstained" if abstained else "unanswerable_candidate_generated"
        elif abstained:
            triage = "model_abstained_on_answerable"
        elif raw_em == 1:
            triage = "exact_match_blocked" if metric["predicted_refused"] else "exact_match_delivered"
        else:
            triage = "inspect_generation_and_answer_format"
        rows.append({
            "id": item["id"], "doc_id": item.get("doc_id"), "category": item.get("category"),
            "question": item["question"], "gold_answer": item["answer"],
            "gold_answerable": item["answerable"], "gold_pages": gold_pages,
            "retrieved_pages": pages, "gold_page_recall": recall,
            "raw_answer": raw, "delivered_answer": prediction.get("answer"),
            "raw_lexical_em": raw_em,
            "raw_lexical_f1": round(token_f1(raw, item["answer"]), 4) if raw_em is not None else None,
            "final_em": metric["em"], "refused": metric["predicted_refused"],
            "refusal_reason": prediction.get("refusal_reason") or (prediction.get("refusal") or {}).get("refusal_reason"),
            "raw_cited_pages": extract_cited_pages(raw or ""),
            "citation_repaired": grounding.get("citation_repaired"),
            "verification_results": grounding.get("verification_results", []),
            "triage": triage, "latency_seconds": prediction.get("latency_seconds"),
            "timings": prediction.get("timings", {}),
            "peak_cuda_allocated_mb": prediction.get("peak_cuda_allocated_mb"),
            "manual_review": {"raw_answer_correct": None, "evidence_sufficient": None,
                              "refusal_appropriate": None, "notes": ""},
        })
    def values(key):
        return [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    times, peaks = values("latency_seconds"), values("peak_cuda_allocated_mb")
    summary = {
        "count": len(rows), "expected_count": len(items),
        "all_gold_pages_retrieved": sum(r["gold_page_recall"] == 1 for r in rows),
        "page_labeled_count": sum(r["gold_page_recall"] is not None for r in rows),
        "refused_count": sum(r["refused"] for r in rows),
        "raw_exact_matches": sum(r["raw_lexical_em"] == 1 for r in rows),
        "exact_matches_blocked": sum(r["raw_lexical_em"] == 1 and r["refused"] for r in rows),
        "refusal_reasons": dict(Counter(r["refusal_reason"] for r in rows if r["refusal_reason"])),
        "triage_counts": dict(Counter(r["triage"] for r in rows)),
        "mean_latency_seconds": round(statistics.mean(times), 4) if times else None,
        "median_latency_seconds": round(statistics.median(times), 4) if times else None,
        "max_peak_cuda_allocated_mb": max(peaks) if peaks else None,
        "total_ingest_seconds": round(sum(r["timings"].get("ingest_seconds", 0) for r in rows), 4),
    }
    return {"status": run.get("status", "unknown"), "summary": summary,
            "note": "Triage labels are review hints, not causal conclusions. Raw EM/F1 require manual review for semantic correctness; CUDA allocated memory excludes non-PyTorch allocations.",
            "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).resolve() in {Path(args.results).resolve(), Path(args.dataset).resolve()}:
        parser.error("Output must not overwrite results or gold dataset")
    run = json.loads(Path(args.results).read_text(encoding="utf-8"))
    expected_hash = run.get("provenance", {}).get("dataset_sha256")
    if expected_hash and expected_hash != file_sha256(args.dataset):
        raise ValueError("Dataset hash differs from the evaluated dataset; use its original snapshot")
    report = diagnose(run, load_eval_dataset(args.dataset))
    report["results_sha256"] = file_sha256(args.results)
    report["dataset_sha256"] = file_sha256(args.dataset)
    atomic_json(args.output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
