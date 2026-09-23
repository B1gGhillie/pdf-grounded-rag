"""Unit tests for the retry loop logic in src.pipeline.rag_retry.

These tests mock src.pipeline.rag_pipeline.run_rag directly so we can
exercise the retry decision logic (_should_retry, _adjust_retrieval_params,
run_rag_with_retry) without depending on a real PDF, index, or model.
"""

import copy
from unittest.mock import patch

from src.eval.ablation import apply_ablation_config
from src.pipeline.rag_retry import (
    _adjust_retrieval_params,
    _should_retry,
    run_rag_with_retry,
)


def _base_cfg(max_retry=1):
    return {
        "retrieval": {"top_k": 3, "fusion": {}},
        "grounding": {"enabled": True, "verify": True},
        "refusal": {"enabled": True, "use_grounding_gate": True},
        "retry": {
            "max_retry": max_retry,
            "strategy": "increase_top_k",
            "top_k_increment": 2,
            "retryable_reasons": ["grounding_failed"],
        },
    }


def _grounding_failed_result():
    return {
        "answer": None,
        "answerable": False,
        "refused": True,
        "refusal_reason": "grounding_failed",
        "refusal": {"refused": True, "refusal_reason": "grounding_failed"},
        "grounding": {
            "grounding_verified": False,
            "failed_citations": [3],
            "feedback": "Page 3: very low lexical overlap (0.05)",
        },
    }


def _success_result():
    return {
        "answer": "The answer [p.1]",
        "answerable": True,
        "refused": False,
        "refusal_reason": None,
        "refusal": {"refused": False, "refusal_reason": None},
        "grounding": {
            "grounding_verified": True,
            "failed_citations": [],
            "feedback": None,
        },
    }


def test_should_retry_on_grounding_failed_with_feedback():
    result = _grounding_failed_result()
    retry_cfg = {"retryable_reasons": ["grounding_failed"]}
    assert _should_retry(result, retry_cfg) is True


def test_should_not_retry_when_not_refused():
    result = _success_result()
    retry_cfg = {"retryable_reasons": ["grounding_failed"]}
    assert _should_retry(result, retry_cfg) is False


def test_should_not_retry_on_non_retryable_reason():
    result = _grounding_failed_result()
    result["refusal_reason"] = "retrieval_miss"
    result["refusal"]["refusal_reason"] = "retrieval_miss"
    retry_cfg = {"retryable_reasons": ["grounding_failed"]}
    assert _should_retry(result, retry_cfg) is False


def test_should_not_retry_without_feedback():
    result = _grounding_failed_result()
    result["grounding"]["feedback"] = None
    result["grounding"]["failed_citations"] = []
    retry_cfg = {"retryable_reasons": ["grounding_failed"]}
    assert _should_retry(result, retry_cfg) is False


def test_adjust_retrieval_params_increases_top_k():
    cfg = _base_cfg(max_retry=2)
    grounding = _grounding_failed_result()["grounding"]
    adjusted = _adjust_retrieval_params(copy.deepcopy(cfg), attempt=1, grounding_result=grounding)
    assert adjusted["retrieval"]["top_k"] == 3 + 2 * 1

    adjusted2 = _adjust_retrieval_params(copy.deepcopy(cfg), attempt=2, grounding_result=grounding)
    assert adjusted2["retrieval"]["top_k"] == 3 + 2 * 2


def test_run_rag_with_retry_disabled_calls_run_rag_once():
    cfg = _base_cfg(max_retry=0)
    with patch("src.pipeline.rag_retry.run_rag") as mock_run_rag:
        mock_run_rag.return_value = _success_result()
        result = run_rag_with_retry("fake.pdf", "question?", cfg, method="full")

    mock_run_rag.assert_called_once()
    assert result["answerable"] is True
    # retry disabled path returns the raw run_rag result unmodified (no retry_metadata)
    assert "retry_metadata" not in result


def test_run_rag_with_retry_succeeds_on_second_attempt():
    cfg = _base_cfg(max_retry=2)
    with patch("src.pipeline.rag_retry.run_rag") as mock_run_rag:
        mock_run_rag.side_effect = [_grounding_failed_result(), _success_result()]
        result = run_rag_with_retry("fake.pdf", "question?", cfg, method="full")

    assert mock_run_rag.call_count == 2
    assert result["answerable"] is True
    assert result["retry_metadata"]["attempt"] == 1
    assert result["retry_metadata"]["retry_triggered"] is False


def test_run_rag_with_retry_exhausts_and_returns_last_failure():
    cfg = _base_cfg(max_retry=2)
    with patch("src.pipeline.rag_retry.run_rag") as mock_run_rag:
        mock_run_rag.return_value = _grounding_failed_result()
        result = run_rag_with_retry("fake.pdf", "question?", cfg, method="full")

    # attempts 0, 1, 2 => 3 calls total (initial + 2 retries)
    assert mock_run_rag.call_count == 3
    assert result["answerable"] is False
    assert result["retry_metadata"]["retry_exhausted"] is True
    assert result["retry_metadata"]["final_attempt"] == 2


def test_run_rag_with_retry_stops_on_non_retryable_reason():
    cfg = _base_cfg(max_retry=2)
    non_retryable = _grounding_failed_result()
    non_retryable["refusal_reason"] = "retrieval_miss"
    non_retryable["refusal"]["refusal_reason"] = "retrieval_miss"

    with patch("src.pipeline.rag_retry.run_rag") as mock_run_rag:
        mock_run_rag.return_value = non_retryable
        result = run_rag_with_retry("fake.pdf", "question?", cfg, method="full")

    # Should stop after the first attempt since retrieval_miss is not retryable
    assert mock_run_rag.call_count == 1
    assert result["retry_metadata"]["retry_triggered"] is False


def test_retry_config_in_ablation():
    """Ablation configs with retry section should apply correctly."""
    base_cfg = {
        "retrieval": {"top_k": 5, "fusion": {}},
        "grounding": {"enabled": False},
        "refusal": {"enabled": False},
    }
    cfg = apply_ablation_config(base_cfg, "full_with_retry")
    assert cfg["retry"]["max_retry"] == 2
    assert cfg["retry"]["retryable_reasons"] == ["grounding_failed"]
    assert cfg["grounding"]["enabled"] is True
    assert cfg["refusal"]["use_grounding_gate"] is True


def test_retry_all_signals_ablation_config():
    base_cfg = {
        "retrieval": {"top_k": 5, "fusion": {}},
        "grounding": {"enabled": False},
        "refusal": {"enabled": False},
    }
    cfg = apply_ablation_config(base_cfg, "retry_all_signals")
    assert set(cfg["retry"]["retryable_reasons"]) == {
        "grounding_failed",
        "retrieval_miss",
        "no_citations",
    }
