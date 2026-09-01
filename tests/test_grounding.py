"""Unit tests for citation parsing and grounding helpers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.grounding.citation import (
    extract_cited_pages,
    is_unanswerable,
    strip_citations,
    validate_citations_in_retrieval,
)
from src.grounding.pipeline import apply_grounding


def test_extract_and_strip_citations():
    text = "Hybrid RAG reached 87.3% [p.3] with recall 0.84 [p.3]."
    assert extract_cited_pages(text) == [3]
    assert "87.3%" in strip_citations(text)
    assert "[p." not in strip_citations(text)


def test_unanswerable_markers():
    assert is_unanswerable("UNANSWERABLE")
    assert is_unanswerable("unanswerable: not in document")
    assert not is_unanswerable("The answer is 87.3% [p.3]")


def test_validate_citations_against_retrieval():
    hits = [{"page_index": 2, "page_number": 3, "fused_score": 0.8}]
    ok, invalid = validate_citations_in_retrieval([3], hits)
    assert ok and invalid == []
    ok, invalid = validate_citations_in_retrieval([9], hits)
    assert not ok and invalid == [9]


def test_apply_grounding_no_citations():
    hits = [{"page_index": 0, "page_number": 1, "page_path": "p.png", "fused_score": 0.9}]
    result = apply_grounding(
        model=None,
        processor=None,
        raw_answer="Something without citations.",
        retrieved_hits=hits,
        grounding_cfg={"verify": True},
    )
    assert result["verification_reason"] == "no_citations"
    assert result["answerable"] is True
    assert result["citations"] == []


def test_apply_grounding_model_refusal():
    result = apply_grounding(
        None,
        None,
        "UNANSWERABLE",
        [{"page_index": 0, "page_number": 1}],
        {"verify": True},
    )
    assert result["answerable"] is False
    assert result["verification_reason"] == "model_refusal"
