"""CPU-friendly Cite-then-Verify surrogate using lexical overlap with page text."""

from __future__ import annotations

from src.grounding.citation import page_number_from_hit
from src.retrieval.bm25_retriever import tokenize


def _overlap(claim_tokens, page_tokens):
    if not claim_tokens:
        return 0.0
    claim = set(claim_tokens)
    page = set(page_tokens)
    if not claim:
        return 0.0
    return len(claim & page) / len(claim)


def verify_grounded_answer_lexical(
    answer,
    cited_pages,
    retrieved_hits,
    verification_threshold=0.5,
    min_token_overlap=0.25,
):
    """Verify citations by checking claim tokens appear in retrieved page snippets."""
    if not cited_pages:
        return {
            "grounding_verified": False,
            "confidence": 0.0,
            "verification_results": [],
            "verification_reason": "no_citations",
        }

    hit_by_page = {page_number_from_hit(h): h for h in retrieved_hits}
    claim_tokens = [
        t
        for t in tokenize(answer)
        if t
        not in {
            "the",
            "a",
            "an",
            "of",
            "to",
            "and",
            "is",
            "are",
            "in",
            "on",
            "for",
            "this",
            "that",
        }
    ]

    results = []
    passed = 0
    for page in cited_pages:
        hit = hit_by_page.get(page)
        if hit is None:
            results.append(
                {
                    "page": page,
                    "supported": False,
                    "raw_response": "PAGE_NOT_IN_RETRIEVAL",
                    "reason": "page_not_retrieved",
                    "overlap": 0.0,
                }
            )
            continue
        snippet = hit.get("text_snippet") or hit.get("text") or ""
        overlap = _overlap(claim_tokens, tokenize(snippet))
        supported = overlap >= min_token_overlap
        results.append(
            {
                "page": page,
                "supported": supported,
                "raw_response": "LEXICAL_YES" if supported else "LEXICAL_NO",
                "reason": "verified" if supported else "evidence_not_found",
                "overlap": round(overlap, 4),
            }
        )
        if supported:
            passed += 1

    confidence = passed / len(cited_pages)
    grounding_verified = confidence >= verification_threshold and passed > 0
    reason = "verified" if grounding_verified else "verification_failed"
    if any(r.get("reason") == "page_not_retrieved" for r in results):
        reason = "citation_outside_retrieval"

    return {
        "grounding_verified": grounding_verified,
        "confidence": round(confidence, 4),
        "verification_results": results,
        "verification_reason": reason,
    }
