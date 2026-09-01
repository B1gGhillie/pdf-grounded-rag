"""Combine refusal signals into a structured decision."""

from src.grounding.citation import is_unanswerable
from src.refusal.score_gate import max_retrieval_score, score_margin, should_refuse_retrieval

REFUSAL_MESSAGES = {
    "retrieval_miss": "No relevant information found in the document.",
    "insufficient_evidence": "Retrieved pages are insufficient to answer.",
    "grounding_failed": "Answer could not be verified against cited pages.",
    "no_citations": "Answer lacks required page citations.",
    "model_refusal": "Model determined the question is unanswerable from the document.",
    "out_of_scope": "Question is out of document scope.",
}


def build_refusal_response(reason, retrieval_score=None, extra=None):
    payload = {
        "answerable": False,
        "answer": None,
        "refused": True,
        "refusal_reason": reason,
        "message": REFUSAL_MESSAGES.get(reason, "Unable to answer this question."),
        "max_retrieval_score": round(retrieval_score, 4) if retrieval_score is not None else None,
    }
    if extra:
        payload.update(extra)
    return payload


def apply_refusal(retrieved_hits, raw_answer=None, grounding_result=None, refusal_cfg=None):
    """Multi-signal refusal: retrieval / model / citations / grounding verify."""
    refusal_cfg = refusal_cfg or {}
    if not refusal_cfg.get("enabled", False):
        return {"refused": False, "refusal_reason": None, "signals": {}}

    threshold = refusal_cfg.get("score_threshold", 0.3)
    margin_threshold = refusal_cfg.get("margin_threshold")
    use_retrieval_gate = refusal_cfg.get("use_retrieval_gate", True)
    use_model_refusal = refusal_cfg.get("use_model_refusal", True)
    use_grounding_gate = refusal_cfg.get("use_grounding_gate", True)
    use_citation_gate = refusal_cfg.get("use_citation_gate", True)

    retrieval_score = max_retrieval_score(retrieved_hits)
    margin = score_margin(retrieved_hits)

    signals = {
        "retrieval_miss": use_retrieval_gate
        and should_refuse_retrieval(retrieved_hits, threshold, margin_threshold),
        "model_refusal": use_model_refusal
        and raw_answer is not None
        and is_unanswerable(raw_answer),
        "no_citations": False,
        "grounding_failed": False,
    }

    if grounding_result is not None:
        answerable = grounding_result.get("answerable", True)
        verified = grounding_result.get("grounding_verified", False)
        has_citations = bool(grounding_result.get("citations"))
        reason = grounding_result.get("verification_reason")

        if use_citation_gate:
            signals["no_citations"] = answerable and reason == "no_citations"

        if use_grounding_gate:
            signals["grounding_failed"] = (
                answerable
                and not verified
                and has_citations
                and reason in ("verification_failed", "citation_outside_retrieval")
            )

    # Priority: retrieval → model → missing citations → failed verify
    if signals["retrieval_miss"]:
        reason = "retrieval_miss"
    elif signals["model_refusal"]:
        reason = "model_refusal"
    elif signals["no_citations"]:
        reason = "no_citations"
    elif signals["grounding_failed"]:
        reason = "grounding_failed"
    else:
        return {
            "refused": False,
            "refusal_reason": None,
            "signals": signals,
            "max_retrieval_score": round(retrieval_score, 4),
            "score_margin": round(margin, 4),
        }

    return {
        "refused": True,
        "refusal_reason": reason,
        "signals": signals,
        "max_retrieval_score": round(retrieval_score, 4),
        "score_margin": round(margin, 4),
        "message": REFUSAL_MESSAGES[reason],
    }
