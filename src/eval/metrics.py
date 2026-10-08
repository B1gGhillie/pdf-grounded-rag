"""Evaluation metrics for QA, retrieval, grounding, and refusal."""

import re
import math
from collections import Counter


def _normalize_text(text):
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r"\[p\.\d+\]", "", text, flags=re.IGNORECASE)
    # Preserve decimal points, signs and units: 0.84 must not equal 84.
    return " ".join(re.findall(r"[+-]?\d+(?:\.\d+)?%?|[a-z]+|[\u4e00-\u9fff]", text))


def token_f1(prediction, reference):
    pred_tokens = _normalize_text(prediction).split()
    ref_tokens = _normalize_text(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    common_count = sum(common.values())
    if not common_count:
        return 0.0
    precision = common_count / len(pred_tokens)
    recall = common_count / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction, reference):
    return int(_normalize_text(prediction) == _normalize_text(reference))


def recall_at_k(retrieved_pages, gold_pages, k):
    """Page recall@k using 1-based page numbers."""
    if not gold_pages:
        return 1.0
    top_k = set(retrieved_pages[:k])
    gold = set(gold_pages)
    return len(top_k & gold) / len(gold)


def precision_at_k(retrieved_pages, gold_pages, k):
    """Fixed-denominator P@k; missing/duplicate retrieved slots score zero."""
    if k <= 0:
        raise ValueError("k must be positive")
    top_k = set(retrieved_pages[:k])
    if not top_k:
        return 0.0
    gold = set(gold_pages)
    return len(top_k & gold) / k


def citation_metrics(cited_pages, gold_pages):
    """Page-level citation precision / recall / F1 vs gold evidence pages."""
    cited = set(cited_pages or [])
    gold = set(gold_pages or [])
    if not cited and not gold:
        return {"citation_precision": 1.0, "citation_recall": 1.0, "citation_f1": 1.0}
    if not cited or not gold:
        return {"citation_precision": 0.0, "citation_recall": 0.0, "citation_f1": 0.0}
    overlap = cited & gold
    precision = len(overlap) / len(cited)
    recall = len(overlap) / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "citation_precision": round(precision, 4),
        "citation_recall": round(recall, 4),
        "citation_f1": round(f1, 4),
    }


def score_answer(prediction, reference):
    return {
        "em": exact_match(prediction, reference),
        "f1": round(token_f1(prediction, reference), 4),
    }


def score_refusal(predicted_refused, gold_answerable):
    gold_refused = not gold_answerable
    return {
        "refusal_correct": int(predicted_refused == gold_refused),
        "predicted_refused": predicted_refused,
        "gold_refused": gold_refused,
        "true_positive_refuse": int(predicted_refused and gold_refused),
        "false_positive_refuse": int(predicted_refused and not gold_refused),
        "false_negative_refuse": int((not predicted_refused) and gold_refused),
        "true_negative_refuse": int((not predicted_refused) and not gold_refused),
    }


def _cited_pages_from_prediction(prediction):
    grounding = prediction.get("grounding") or {}
    citations = grounding.get("citations") or []
    pages = []
    for c in citations:
        if isinstance(c, dict) and "page" in c:
            pages.append(c["page"])
        elif isinstance(c, int):
            pages.append(c)
    if pages:
        return pages
    # Fallback: parse from raw_answer
    raw = prediction.get("raw_answer") or prediction.get("answer") or ""
    return [int(m) for m in re.findall(r"\[p\.(\d+)\]", str(raw), flags=re.IGNORECASE)]


def score_prediction(prediction, gold):
    from src.grounding.citation import is_unanswerable
    gold_answerable = gold.get("answerable", True)
    predicted_refused = bool(prediction.get("refused", False) or not prediction.get("answerable", True)
                             or is_unanswerable(prediction.get("answer")))

    retry_meta = prediction.get("retry_metadata") or {}
    metrics = {
        "id": gold.get("id"),
        "method": prediction.get("method"),
        "doc_id": gold.get("document_sha256", gold.get("pdf_path")),
        "latency_seconds": prediction.get("latency_seconds"),
        "confidence": prediction.get("evidence_confidence", (prediction.get("refusal") or {}).get("max_retrieval_score", 0.0)),
        "refusal_reason": prediction.get("refusal_reason")
        or (prediction.get("refusal") or {}).get("refusal_reason"),
        "retry_attempts": retry_meta.get("attempt"),
        "retry_triggered": int(bool(retry_meta.get("retry_triggered"))) if retry_meta else None,
        "retry_exhausted": int(bool(retry_meta.get("retry_exhausted"))) if retry_meta else None,
    }
    metrics.update(score_refusal(predicted_refused, gold_answerable))

    retrieved = []
    for p in prediction.get("retrieved_pages", []):
        if "page_number" in p:
            retrieved.append(p["page_number"])
        elif "page_index" in p:
            retrieved.append(p["page_index"] + 1)

    gold_pages = gold.get("gold_pages", [])

    # Retrieval metrics use fixed cutoffs.  The previous implementation used
    # the number of returned hits as k, which made a nominal recall@5 depend on
    # how many non-zero BM25 results happened to be returned.
    for cutoff in (1, 3, 5):
        metrics[f"page_recall@{cutoff}"] = round(
            recall_at_k(retrieved, gold_pages, k=cutoff), 4
        ) if gold_pages else None
        metrics[f"page_precision@{cutoff}"] = round(
            precision_at_k(retrieved, gold_pages, k=cutoff), 4
        ) if gold_pages else None
    metrics["page_recall@k"] = metrics["page_recall@5"]
    metrics["page_precision@k"] = metrics["page_precision@5"]

    if gold_answerable:
        answer = prediction.get("answer") or ""
        # A refusal on an answerable item is a zero-answer outcome in the
        # overall score.  Keep conditional metrics separate for coverage
        # analysis.
        if predicted_refused:
            metrics["em"] = 0
            metrics["f1"] = 0.0
            metrics["conditional_em"] = None
            metrics["conditional_f1"] = None
        else:
            answer_scores = score_answer(answer, gold.get("answer", ""))
            metrics.update(answer_scores)
            metrics["conditional_em"] = answer_scores["em"]
            metrics["conditional_f1"] = answer_scores["f1"]
        # Score citations actually delivered, not a rejected internal draft.
        metrics.update(citation_metrics([] if predicted_refused else _cited_pages_from_prediction(prediction), gold_pages))
        grounding = prediction.get("grounding") or {}
        if grounding:
            verified = grounding.get("grounding_verified")
            metrics["grounding_verified"] = int(verified) if verified is not None else None
            metrics["grounding_confidence"] = grounding.get("confidence")
        else:
            metrics["grounding_verified"] = None
            metrics["grounding_confidence"] = None
    else:
        metrics["em"] = None
        metrics["f1"] = None
        metrics["conditional_em"] = None
        metrics["conditional_f1"] = None
        metrics["citation_precision"] = None
        metrics["citation_recall"] = None
        metrics["citation_f1"] = None
        metrics["grounding_verified"] = None
        metrics["grounding_confidence"] = None

    metrics["end_to_end_correct"] = int(predicted_refused) if not gold_answerable else metrics["em"]
    metrics["answered_correct"] = None if predicted_refused else (metrics["em"] if gold_answerable else 0)
    raw = prediction.get("raw_answer")
    metrics["candidate_correct"] = (exact_match(raw, gold.get("answer", "")) if gold_answerable else 0) if raw and not is_unanswerable(raw) else None
    metrics["unsupported_answer"] = int(not gold_answerable and not predicted_refused)
    return metrics


def _safe_div(num, den):
    return round(num / den, 4) if den else None


def aggregate_metrics(per_item_metrics):
    def _avg(key):
        vals = [m[key] for m in per_item_metrics if m.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    tp = sum(m.get("true_positive_refuse", 0) for m in per_item_metrics)
    fp = sum(m.get("false_positive_refuse", 0) for m in per_item_metrics)
    fn = sum(m.get("false_negative_refuse", 0) for m in per_item_metrics)

    refusal_precision = _safe_div(tp, tp + fp)
    refusal_recall = _safe_div(tp, tp + fn)
    refusal_f1 = _safe_div(2 * tp, 2 * tp + fp + fn)

    reason_counts = {}
    for m in per_item_metrics:
        reason = m.get("refusal_reason")
        if reason:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    return {
        "count": len(per_item_metrics),
        "em": _avg("em"),
        "f1": _avg("f1"),
        "end_to_end_accuracy": _avg("end_to_end_correct"),
        "coverage": _safe_div(sum(not m.get("predicted_refused") for m in per_item_metrics), len(per_item_metrics)),
        "selective_accuracy": _avg("answered_correct"),
        "selective_risk": round(1 - _avg("answered_correct"), 4) if _avg("answered_correct") is not None else None,
        "unanswerable_answer_rate": _safe_div(sum(m.get("unsupported_answer", 0) for m in per_item_metrics), sum(bool(m.get("gold_refused")) for m in per_item_metrics)),
        "mean_latency_seconds": _avg("latency_seconds"),
        "conditional_em": _avg("conditional_em"),
        "conditional_f1": _avg("conditional_f1"),
        "answer_coverage": _safe_div(
            sum(1 for m in per_item_metrics if m.get("gold_refused") is False and not m.get("predicted_refused")),
            sum(1 for m in per_item_metrics if m.get("gold_refused") is False),
        ),
        "answerable_count": sum(1 for m in per_item_metrics if m.get("gold_refused") is False),
        "answered_answerable": sum(
            1 for m in per_item_metrics
            if m.get("gold_refused") is False and not m.get("predicted_refused")
        ),
        "page_recall@k": _avg("page_recall@k"),
        "page_precision@k": _avg("page_precision@k"),
        "page_recall@1": _avg("page_recall@1"),
        "page_recall@3": _avg("page_recall@3"),
        "page_recall@5": _avg("page_recall@5"),
        "page_precision@1": _avg("page_precision@1"),
        "page_precision@3": _avg("page_precision@3"),
        "page_precision@5": _avg("page_precision@5"),
        "citation_precision": _avg("citation_precision"),
        "citation_recall": _avg("citation_recall"),
        "citation_f1": _avg("citation_f1"),
        "grounding_verified_rate": _avg("grounding_verified"),
        "refusal_accuracy": _avg("refusal_correct"),
        "refusal_precision": refusal_precision,
        "refusal_recall": refusal_recall,
        "refusal_f1": refusal_f1,
        "refusal_reason_counts": reason_counts,
        "avg_retry_attempts": _avg("retry_attempts"),
        "retry_triggered_rate": _avg("retry_triggered"),
        "retry_exhausted_rate": _avg("retry_exhausted"),
    }


def risk_coverage_curve(metrics):
    """Empirical curve for available generated candidates, with ties grouped.

    Pre-generation refusals have no observable answer correctness and are
    excluded explicitly. This is not a full counterfactual pipeline curve.
    """
    rows = [m for m in metrics if m.get("candidate_correct") is not None]
    rows = [m for m in rows if math.isfinite(float(m.get("confidence", 0)))]
    result, total, correct = [], 0, 0
    for threshold in sorted({m["confidence"] for m in rows}, reverse=True):
        group = [m for m in rows if m["confidence"] == threshold]
        total += len(group)
        correct += sum(m["candidate_correct"] for m in group)
        result.append({"threshold": threshold, "coverage": total/len(rows), "risk": 1-correct/total})
    return {"scope": "observed_generated_candidates", "count": len(rows), "total_count": len(metrics), "points": result}


def document_bootstrap_ci(metrics, keys=("f1", "end_to_end_correct"), samples=1000, seed=42):
    """Resample PDF clusters, not correlated questions from the same PDF."""
    import random
    clusters = {}
    for row in metrics:
        clusters.setdefault(row.get("doc_id"), []).append(row)
    if len(clusters) < 2:
        return {"method": "document_cluster_bootstrap", "document_count": len(clusters),
                "intervals": None, "reason": "at_least_two_independent_documents_required"}
    rng = random.Random(seed)
    groups = list(clusters.values())
    draws = {key: [] for key in keys}
    for _ in range(samples):
        rows = [r for g in rng.choices(groups, k=len(groups)) for r in g]
        for key in keys:
            values = [r[key] for r in rows if r.get(key) is not None]
            if values:
                draws[key].append(sum(values)/len(values))
    intervals = {}
    for key, values in draws.items():
        values.sort()
        intervals[key] = [values[int(.025*(len(values)-1))], values[int(.975*(len(values)-1))]] if values else None
    return {"method": "document_cluster_bootstrap_percentile_95", "document_count": len(clusters),
            "samples": samples, "seed": seed, "intervals": intervals}
