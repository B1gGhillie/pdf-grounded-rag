"""Page citation extraction and validation for grounded answers."""

import re

CITATION_RE = re.compile(r"\[p\.(\d+)\]", re.IGNORECASE)
UNANSWERABLE_MARKERS = ("UNANSWERABLE", '"answerable": false', '"answerable":false')


def page_number_from_hit(hit):
    """Resolve 1-based document page number from a retrieval hit."""
    if "page_number" in hit:
        return hit["page_number"]
    return hit["page_index"] + 1


def is_unanswerable(answer):
    """Detect legacy UNANSWERABLE or structured refusal in model output."""
    text = answer.strip()
    upper = text.upper()
    if upper == "UNANSWERABLE" or upper.startswith("UNANSWERABLE"):
        return True
    return any(marker in text for marker in UNANSWERABLE_MARKERS)


def extract_cited_pages(answer):
    """Return unique 1-based page numbers cited in answer, in order of appearance."""
    seen = set()
    pages = []
    for match in CITATION_RE.finditer(answer):
        page = int(match.group(1))
        if page not in seen:
            seen.add(page)
            pages.append(page)
    return pages


def strip_citations(answer):
    """Remove [p.N] markers and collapse extra whitespace."""
    cleaned = CITATION_RE.sub("", answer)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def retrieved_page_numbers(retrieved_hits):
    return {page_number_from_hit(hit) for hit in retrieved_hits}


def page_path_by_number(retrieved_hits):
    """Map 1-based page number to image path."""
    mapping = {}
    for hit in retrieved_hits:
        page_num = page_number_from_hit(hit)
        if hit.get("page_path"):
            mapping[page_num] = hit["page_path"]
    return mapping


def validate_citations_in_retrieval(cited_pages, retrieved_hits):
    """Check whether every cited page appears in the retrieved evidence set."""
    allowed = retrieved_page_numbers(retrieved_hits)
    invalid = [page for page in cited_pages if page not in allowed]
    return len(invalid) == 0, invalid


def build_citation_records(cited_pages, retrieved_hits):
    """Attach retrieval metadata to each cited page."""
    hit_by_page = {page_number_from_hit(hit): hit for hit in retrieved_hits}
    records = []

    for page in cited_pages:
        hit = hit_by_page.get(page, {})
        score = hit.get("fused_score", hit.get("score"))
        records.append(
            {
                "page": page,
                "page_index": hit.get("page_index"),
                "retrieval_score": round(score, 4) if score is not None else None,
                "in_retrieved_set": page in hit_by_page,
                "text_snippet": hit.get("text_snippet", ""),
            }
        )
    return records
