"""Evaluation metrics for QA, retrieval, grounding, and refusal."""

import re
import string


def _normalize_text(text):
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r"\[p\.\d+\]", "", text, flags=re.IGNORECASE)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def token_f1(prediction, reference):
    pred_tokens = _normalize_text(prediction).split()
    ref_tokens = _normalize_text(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
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
    """Page precision@k using 1-based page numbers."""
    if not retrieved_pages:
        return 0.0 if gold_pages else 1.0
    top_k = set(retrieved_pages[:k])
    if not top_k:
        return 0.0
    gold = set(gold_pages)
    return len(top_k & gold) / len(top_k)


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
    gold_answerable = gold.get("answerable", True)
    predicted_refused = prediction.get("refused", False) or not prediction.get("answerable", True)

    retry_meta = prediction.get("retry_metadata") or {}
    metrics = {
        "id": gold.get("id"),
        "method": prediction.get("method"),
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

    k = len(retrieved) or 1
    gold_pages = gold.get("gold_pages", [])

    if gold_answerable and not predicted_refused:
        answer = prediction.get("answer") or ""
        metrics.update(score_answer(answer, gold.get("answer", "")))
        metrics["page_recall@k"] = round(recall_at_k(retrieved, gold_pages, k=k), 4)
        metrics["page_precision@k"] = round(precision_at_k(retrieved, gold_pages, k=k), 4)
        metrics.update(citation_metrics(_cited_pages_from_prediction(prediction), gold_pages))
        grounding = prediction.get("grounding") or {}
        if grounding:
            metrics["grounding_verified"] = int(bool(grounding.get("grounding_verified")))
            metrics["grounding_confidence"] = grounding.get("confidence")
        else:
            metrics["grounding_verified"] = None
            metrics["grounding_confidence"] = None
    else:
        metrics["em"] = None
        metrics["f1"] = None
        metrics["page_recall@k"] = (
            round(recall_at_k(retrieved, gold_pages, k=k), 4) if gold_pages else None
        )
        metrics["page_precision@k"] = None
        metrics["citation_precision"] = None
        metrics["citation_recall"] = None
        metrics["citation_f1"] = None
        metrics["grounding_verified"] = None
        metrics["grounding_confidence"] = None

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
    if refusal_precision is not None and refusal_recall is not None and (refusal_precision + refusal_recall):
        refusal_f1 = round(
            2 * refusal_precision * refusal_recall / (refusal_precision + refusal_recall), 4
        )
    else:
        refusal_f1 = None

    reason_counts = {}
    for m in per_item_metrics:
        reason = m.get("refusal_reason")
        if reason:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    return {
        "count": len(per_item_metrics),
        "em": _avg("em"),
        "f1": _avg("f1"),
        "page_recall@k": _avg("page_recall@k"),
        "page_precision@k": _avg("page_precision@k"),
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
