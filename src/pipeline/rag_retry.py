"""Failure-aware bounded retrieval retries, with immutable per-question config."""
from __future__ import annotations
import copy
import logging
from time import perf_counter
logger = logging.getLogger(__name__)


def run_rag(*args, **kwargs):
    from src.pipeline.rag_pipeline import run_rag as impl
    return impl(*args, **kwargs)


def _should_retry(result, retry_cfg):
    if not result.get("refused"):
        return False
    reason = result.get("refusal_reason") or result.get("refusal", {}).get("refusal_reason")
    if reason not in retry_cfg.get("retryable_reasons", ["grounding_failed"]):
        return False
    if reason in {"retrieval_miss", "no_citations"}:
        return True
    grounding = result.get("grounding") or {}
    return bool(grounding.get("feedback") or grounding.get("failed_citations"))


def _adjust_retrieval_params(cfg, attempt, grounding_result):
    cfg = copy.deepcopy(cfg)
    retry = cfg.get("retry", {})
    strategy = retry.get("strategy", "increase_top_k")
    if strategy in {"increase_top_k", "evidence_budget"}:
        cfg["retrieval"]["top_k"] = min(
            cfg["retrieval"].get("top_k", 5) + retry.get("top_k_increment", 2)*attempt,
            retry.get("max_top_k", 9))
    elif strategy == "adjust_fusion":
        fusion = cfg["retrieval"].setdefault("fusion", {})
        fusion["min_text_weight"], fusion["max_text_weight"] = 0.4, 0.8
    else:
        raise ValueError(f"Unknown retry strategy: {strategy}")
    return cfg


def run_rag_with_retry(pdf_path, question, cfg, method="full", force_rebuild=False):
    started = perf_counter()
    base = copy.deepcopy(cfg)
    retry = base.get("retry", {})
    limit = retry.get("max_retry", 0)
    if not isinstance(limit, int) or limit < 0:
        raise ValueError("max_retry must be a non-negative integer")
    if limit == 0:
        return run_rag(pdf_path, question, base, method=method, force_rebuild=force_rebuild)
    if base.get("refusal", {}).get("calibration_path"):
        raise ValueError("Adaptive retries need separately validated thresholds per retrieval setting; use max_retry=0 with a single calibration artifact")
    if retry.get("max_total_pages", 30) < base["retrieval"]["top_k"]:
        raise ValueError("max_total_pages must cover the initial top_k")
    if retry.get("max_top_k", 9) < base["retrieval"]["top_k"] or retry.get("top_k_increment", 2) < 0:
        raise ValueError("Retry top-k limits cannot shrink the initial evidence set")
    history, seen, spent = [], set(), 0
    current = copy.deepcopy(base)
    result = None
    stop = "complete"
    for attempt in range(limit+1):
        result = run_rag(pdf_path, question, current, method=method,
                         force_rebuild=force_rebuild and attempt == 0)
        grounding = result.get("grounding") or {}
        pages = tuple(h.get("page_number", h.get("page_index", -1)+1)
                      for h in result.get("retrieved_pages", []))
        spent += len(pages)
        history.append({"attempt": attempt, "top_k": current["retrieval"]["top_k"],
                        "pages": list(pages), "refused": bool(result.get("refused")),
                        "reason": result.get("refusal_reason"), "feedback": grounding.get("feedback"),
                        "latency_seconds": result.get("latency_seconds"),
                        "peak_cuda_allocated_mb": result.get("peak_cuda_allocated_mb"),
                        "failed_citations": grounding.get("failed_citations", [])})
        if not _should_retry(result, retry):
            stop = "accepted" if not result.get("refused") else "non_retryable"
            break
        if attempt == limit:
            stop = "max_retry"
            break
        reason = result.get("refusal_reason")
        state = (pages, reason, result.get("raw_answer"))
        if retry.get("strategy") == "evidence_budget" and state in seen:
            stop = "unchanged_evidence"
            break
        seen.add(state)
        current = _adjust_retrieval_params(base, attempt+1, grounding)
        if retry.get("strategy") == "evidence_budget" and reason == "no_citations":
            current["retrieval"]["top_k"] = history[-1]["top_k"]
        if spent + current["retrieval"]["top_k"] > retry.get("max_total_pages", 30):
            stop = "page_budget"
            break
        current.setdefault("generation", {})["retry_feedback"] = (
            grounding.get("feedback") or f"Previous attempt failed: {reason}. Return an answer with valid citations or UNANSWERABLE."
        )
    result["retry_metadata"] = {
        "attempt": len(history)-1, "final_attempt": len(history)-1, "max_retry": limit,
        "retry_triggered": len(history) > 1,
        "retry_exhausted": bool(result.get("refused")) and stop in {"max_retry", "page_budget", "unchanged_evidence"},
        "stop_reason": stop, "total_evidence_pages": spent, "history": history,
    }
    result["latency_seconds"] = round(perf_counter()-started, 6)
    peaks = [h["peak_cuda_allocated_mb"] for h in history if h.get("peak_cuda_allocated_mb") is not None]
    result["peak_cuda_allocated_mb"] = max(peaks) if peaks else None
    return result
