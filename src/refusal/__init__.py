"""Multi-signal refusal: retrieval gate + model refusal + citation + grounding."""

from src.refusal.decision import apply_refusal, build_refusal_response
from src.refusal.score_gate import max_retrieval_score, score_margin, should_refuse_retrieval

__all__ = [
    "apply_refusal",
    "build_refusal_response",
    "max_retrieval_score",
    "score_margin",
    "should_refuse_retrieval",
]
