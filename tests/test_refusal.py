"""Unit tests for multi-signal refusal."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.refusal.decision import apply_refusal
from src.refusal.score_gate import max_retrieval_score, score_margin, should_refuse_retrieval


def test_absolute_score_not_forced_to_one():
    hits = [
        {"fused_score": 0.12},
        {"fused_score": 0.08},
    ]
    # Old bug: min-max within Top-K made max == 1.0
    assert max_retrieval_score(hits) == 0.12
    assert should_refuse_retrieval(hits, threshold=0.25) is True
    assert should_refuse_retrieval(hits, threshold=0.10) is False


def test_score_margin():
    hits = [{"fused_score": 0.9}, {"fused_score": 0.85}]
    assert abs(score_margin(hits) - 0.05) < 1e-9
    assert should_refuse_retrieval(hits, threshold=0.1, margin_threshold=0.1) is True


def test_empty_hits_refuse():
    assert should_refuse_retrieval([], 0.25) is True
    decision = apply_refusal([], refusal_cfg={"enabled": True, "score_threshold": 0.25})
    assert decision["refused"] is True
    assert decision["refusal_reason"] == "retrieval_miss"


def test_model_refusal_signal():
    hits = [{"fused_score": 0.8}]
    decision = apply_refusal(
        hits,
        raw_answer="UNANSWERABLE",
        grounding_result=None,
        refusal_cfg={"enabled": True, "score_threshold": 0.25},
    )
    assert decision["refused"] is True
    assert decision["refusal_reason"] == "model_refusal"


def test_no_citations_gate():
    hits = [{"fused_score": 0.8}]
    grounding = {
        "answerable": True,
        "citations": [],
        "grounding_verified": False,
        "verification_reason": "no_citations",
    }
    decision = apply_refusal(
        hits,
        raw_answer="Some answer without cites",
        grounding_result=grounding,
        refusal_cfg={
            "enabled": True,
            "score_threshold": 0.25,
            "use_citation_gate": True,
            "use_grounding_gate": True,
        },
    )
    assert decision["refused"] is True
    assert decision["refusal_reason"] == "no_citations"


def test_grounding_failed_gate():
    hits = [{"fused_score": 0.8}]
    grounding = {
        "answerable": True,
        "citations": [{"page": 1}],
        "grounding_verified": False,
        "verification_reason": "verification_failed",
    }
    decision = apply_refusal(
        hits,
        raw_answer="Claim [p.1]",
        grounding_result=grounding,
        refusal_cfg={"enabled": True, "score_threshold": 0.25},
    )
    assert decision["refusal_reason"] == "grounding_failed"
