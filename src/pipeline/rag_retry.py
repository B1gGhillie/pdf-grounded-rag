"""RAG pipeline with retry loop for grounding failures (CaVe-VLM-CoT-inspired)."""

from __future__ import annotations

import logging

from src.pipeline.rag_pipeline import run_rag

logger = logging.getLogger(__name__)


def _should_retry(result, retry_cfg):
    """Determine if this result warrants a retry attempt."""
    if not result.get("refused", False):
        return False
    
    refusal_reason = result.get("refusal_reason") or result.get("refusal", {}).get("refusal_reason")
    
    # Only retry grounding failures by default
    retryable_reasons = retry_cfg.get("retryable_reasons", ["grounding_failed"])
    
    if refusal_reason not in retryable_reasons:
        return False
    
    # Need feedback to guide the retry
    grounding = result.get("grounding", {})
    feedback = grounding.get("feedback")
    failed_citations = grounding.get("failed_citations", [])
    
    return bool(feedback or failed_citations)


def _filter_failed_pages(retrieved_hits, failed_pages):
    """Remove pages that failed verification from retrieved set."""
    if not failed_pages:
        return retrieved_hits
    
    failed_set = set(failed_pages)
    filtered = [
        h for h in retrieved_hits
        if h.get("page_number", h.get("page_index", -1) + 1) not in failed_set
    ]
    
    logger.info("Retry: filtered {0} failed pages, {1} pages remain".format(
        len(failed_pages), len(filtered)))
    
    return filtered


def _adjust_retrieval_params(cfg, attempt, grounding_result):
    """Adjust retrieval parameters for retry attempt."""
    retry_cfg = cfg.get("retry", {})
    strategy = retry_cfg.get("strategy", "increase_top_k")
    
    if strategy == "increase_top_k":
        # Expand retrieval to get more candidate pages
        top_k_delta = retry_cfg.get("top_k_increment", 2)
        cfg["retrieval"]["top_k"] = cfg["retrieval"].get("top_k", 5) + top_k_delta * attempt
        logger.info("Retry attempt {0}: increased top_k to {1}".format(
            attempt, cfg["retrieval"]["top_k"]))
    
    elif strategy == "adjust_fusion":
        # Shift text/visual weights based on failure type
        feedback = grounding_result.get("feedback", "")
        fusion_cfg = cfg["retrieval"].setdefault("fusion", {})
        
        if "lexical overlap" in feedback.lower():
            # Low lexical overlap → increase text weight
            fusion_cfg["min_text_weight"] = 0.4
            fusion_cfg["max_text_weight"] = 0.8
            logger.info("Retry attempt {0}: increased text weight (low lexical overlap)".format(attempt))
        else:
            # Generic failure → balance weights
            fusion_cfg["min_text_weight"] = 0.3
            fusion_cfg["max_text_weight"] = 0.7
    
    return cfg


def run_rag_with_retry(pdf_path, question, cfg, method="full", force_rebuild=False):
    """Run RAG with retry loop on grounding failures.
    
    Retry strategy (simplified from CaVe-VLM-CoT):
    1. Run standard RAG pipeline
    2. If grounding_failed with structured feedback:
       - Option A (filter): Remove failed pages and regenerate from remaining
       - Option B (expand): Increase top_k to retrieve more candidates
    3. Repeat until success or max_retry exhausted
    
    Args:
        pdf_path: Document path
        question: User query
        cfg: Config dict (must contain retry section if enabled)
        method: RAG method (full/hybrid/visual/ocr)
        force_rebuild: Force index rebuild
        
    Returns:
        Final result dict with added retry_metadata
    """
    retry_cfg = cfg.get("retry", {})
    max_retry = retry_cfg.get("max_retry", 0)
    
    if max_retry == 0:
        # Retry disabled, run standard path
        return run_rag(pdf_path, question, cfg, method=method, force_rebuild=force_rebuild)
    
    result = None
    for attempt in range(max_retry + 1):
        result = run_rag(pdf_path, question, cfg, method=method, force_rebuild=force_rebuild)
        
        # Add retry metadata
        result.setdefault("retry_metadata", {})
        result["retry_metadata"]["attempt"] = attempt
        result["retry_metadata"]["max_retry"] = max_retry
        
        if attempt == 0:
            result["retry_metadata"]["initial_result"] = True
        
        # Check if we should retry
        if not _should_retry(result, retry_cfg):
            result["retry_metadata"]["retry_triggered"] = False
            if attempt > 0:
                logger.info("Retry attempt {0} succeeded or hit non-retryable state".format(attempt))
            break
        
        if attempt >= max_retry:
            result["retry_metadata"]["retry_exhausted"] = True
            logger.warning("Max retry {0} exhausted, returning last failed result".format(max_retry))
            break
        
        # Extract feedback for next attempt
        grounding = result.get("grounding", {})
        failed_pages = grounding.get("failed_citations", [])
        feedback = grounding.get("feedback", "")
        
        logger.info("Retry attempt {0}: grounding failed, feedback: {1}".format(attempt + 1, feedback))
        
        result["retry_metadata"]["retry_triggered"] = True
        result["retry_metadata"]["retry_reason"] = feedback
        
        # Adjust config for next attempt
        cfg = _adjust_retrieval_params(dict(cfg), attempt + 1, grounding)
        
        # Note: We don't filter pages here because run_rag does a fresh retrieval each time.
        # The increased top_k gives us new candidates to work with.
    
    # Attach retry summary to final result
    if result:
        result["retry_metadata"]["final_attempt"] = result["retry_metadata"]["attempt"]
    
    return result
