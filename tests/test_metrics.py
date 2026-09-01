"""Unit tests for evaluation metrics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.metrics import (
    aggregate_metrics,
    citation_metrics,
    score_prediction,
    token_f1,
)


def test_token_f1_and_citation_metrics():
    assert token_f1("87.3%", "87.3%") == 1.0
    cites = citation_metrics([1, 3], [3])
    assert cites["citation_precision"] == 0.5
    assert cites["citation_recall"] == 1.0


def test_score_prediction_answerable():
    pred = {
        "answer": "87.3%",
        "raw_answer": "Hybrid accuracy is 87.3% [p.3]",
        "refused": False,
        "answerable": True,
        "retrieved_pages": [{"page_index": 2, "page_number": 3}],
        "grounding": {
            "citations": [{"page": 3}],
            "grounding_verified": True,
            "confidence": 1.0,
        },
    }
    gold = {
        "id": "q2",
        "answer": "87.3%",
        "answerable": True,
        "gold_pages": [3],
    }
    m = score_prediction(pred, gold)
    assert m["em"] == 1
    assert m["citation_f1"] == 1.0
    assert m["page_recall@k"] == 1.0


def test_aggregate_refusal_prf():
    items = [
        {
            "refusal_correct": 1,
            "true_positive_refuse": 1,
            "false_positive_refuse": 0,
            "false_negative_refuse": 0,
            "true_negative_refuse": 0,
            "em": None,
            "f1": None,
            "refusal_reason": "model_refusal",
        },
        {
            "refusal_correct": 0,
            "true_positive_refuse": 0,
            "false_positive_refuse": 1,
            "false_negative_refuse": 0,
            "true_negative_refuse": 0,
            "em": None,
            "f1": None,
            "refusal_reason": "retrieval_miss",
        },
    ]
    summary = aggregate_metrics(items)
    assert summary["refusal_precision"] == 0.5
    assert summary["refusal_recall"] == 1.0
    assert summary["refusal_reason_counts"]["model_refusal"] == 1
