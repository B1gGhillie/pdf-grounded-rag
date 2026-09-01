"""Grounding pipeline: parse citations, validate retrieval overlap, verify evidence."""

from src.grounding.citation import (
    build_citation_records,
    extract_cited_pages,
    is_unanswerable,
    strip_citations,
    validate_citations_in_retrieval,
)
from src.grounding.lexical_verifier import verify_grounded_answer_lexical
from src.grounding.verifier import verify_grounded_answer


def apply_grounding(
    model,
    processor,
    raw_answer,
    retrieved_hits,
    grounding_cfg,
):
    """Apply page-level grounding to a generated answer.

    Steps:
    1. Detect unanswerable responses
    2. Extract [p.N] citations from the answer
    3. Validate citations against retrieved pages
    4. Optionally run Cite-then-Verify on cited pages
       (VLM YES/NO when model available; else lexical surrogate)
    """
    if is_unanswerable(raw_answer):
        return {
            "answerable": False,
            "answer": raw_answer,
            "answer_text": None,
            "citations": [],
            "citations_valid": True,
            "invalid_citations": [],
            "grounding_verified": False,
            "confidence": 0.0,
            "verification_results": [],
            "verification_reason": "model_refusal",
        }

    cited_pages = extract_cited_pages(raw_answer)
    citations_valid, invalid_citations = validate_citations_in_retrieval(
        cited_pages, retrieved_hits
    )
    citation_records = build_citation_records(cited_pages, retrieved_hits)
    answer_text = strip_citations(raw_answer)

    result = {
        "answerable": True,
        "answer": raw_answer,
        "answer_text": answer_text,
        "citations": citation_records,
        "citations_valid": citations_valid,
        "invalid_citations": invalid_citations,
        "grounding_verified": False,
        "confidence": 0.0,
        "verification_results": [],
        "verification_reason": None,
    }

    if not cited_pages:
        result["verification_reason"] = "no_citations"
        return result

    if not citations_valid:
        result["verification_reason"] = "citation_outside_retrieval"
        return result

    if not grounding_cfg.get("verify", True):
        result["grounding_verified"] = True
        result["confidence"] = 1.0
        result["verification_reason"] = "verify_disabled"
        return result

    threshold = grounding_cfg.get("verification_threshold", 0.5)
    mode = (grounding_cfg.get("verify_mode") or "auto").lower()

    use_vlm = mode == "vlm" or (mode == "auto" and model is not None and processor is not None)
    if use_vlm:
        verify_result = verify_grounded_answer(
            model,
            processor,
            answer_text or raw_answer,
            cited_pages,
            retrieved_hits,
            verification_threshold=threshold,
        )
    else:
        verify_result = verify_grounded_answer_lexical(
            answer_text or raw_answer,
            cited_pages,
            retrieved_hits,
            verification_threshold=threshold,
            min_token_overlap=grounding_cfg.get("lexical_min_overlap", 0.25),
        )
        verify_result["verification_backend"] = "lexical"

    result.update(verify_result)
    return result
