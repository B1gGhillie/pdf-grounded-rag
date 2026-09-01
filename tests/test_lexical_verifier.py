"""Unit tests for lexical Cite-then-Verify surrogate."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.grounding.lexical_verifier import verify_grounded_answer_lexical
from src.grounding.pipeline import apply_grounding


def test_lexical_verify_passes_on_overlapping_snippet():
    hits = [
        {
            "page_index": 2,
            "page_number": 3,
            "page_path": "p3.png",
            "fused_score": 0.9,
            "text_snippet": "On the sample benchmark, hybrid RAG achieved 87.3% answer accuracy.",
        }
    ]
    result = verify_grounded_answer_lexical(
        "hybrid RAG achieved 87.3% answer accuracy",
        [3],
        hits,
        verification_threshold=0.5,
        min_token_overlap=0.2,
    )
    assert result["grounding_verified"] is True
    assert result["verification_reason"] == "verified"


def test_apply_grounding_lexical_mode_without_vlm():
    hits = [
        {
            "page_index": 0,
            "page_number": 1,
            "page_path": "p1.png",
            "fused_score": 0.8,
            "text_snippet": "The main contribution is page-level grounding.",
        }
    ]
    result = apply_grounding(
        None,
        None,
        "The main contribution is page-level grounding [p.1]",
        hits,
        {"verify": True, "verify_mode": "lexical", "lexical_min_overlap": 0.2},
    )
    assert result["citations_valid"] is True
    assert result["grounding_verified"] is True
