"""Retrieval-score gate for pre-generation refusal.

Uses evidence_confidence (a monotone absolute score, not a probability).
Do NOT re-normalize within Top-K: min-max over Top-K forces max≈1.0 whenever
any hit exists, which disables the retrieval gate.
"""
import math


def _hit_score(hit):
    value = float(hit.get("evidence_confidence", hit.get("fused_score", hit.get("score", 0.0))))
    if not math.isfinite(value):
        raise ValueError("Non-finite evidence confidence")
    return value


def max_retrieval_score(retrieved_hits):
    """Return absolute max retrieval confidence (typically in [0, 1] after fusion)."""
    if not retrieved_hits:
        return 0.0
    return max(_hit_score(h) for h in retrieved_hits)


def score_margin(retrieved_hits):
    """Top-1 minus Top-2 score gap; low margin indicates retrieval ambiguity."""
    if not retrieved_hits:
        return 0.0
    scores = sorted((_hit_score(h) for h in retrieved_hits), reverse=True)
    if len(scores) == 1:
        return scores[0]
    return scores[0] - scores[1]


def should_refuse_retrieval(retrieved_hits, threshold, margin_threshold=None):
    """Refuse when retrieval confidence is below threshold (and optionally low margin)."""
    if not retrieved_hits:
        return True
    max_score = max_retrieval_score(retrieved_hits)
    if max_score < threshold:
        return True
    if margin_threshold is not None and score_margin(retrieved_hits) < margin_threshold:
        return True
    return False
