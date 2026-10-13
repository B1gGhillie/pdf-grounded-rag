"""Unified RAG pipeline: retrieval, refusal, generation, grounding."""

from __future__ import annotations

import gc
import logging
import re
import copy
import json
from time import perf_counter
from pathlib import Path

from src.generation.extractive_generator import answer_extractive_with_citations
from src.generation.vlm_generator import (
    answer_with_grounding,
    answer_with_text_context,
    answer_with_vlm,
    load_vlm,
)
from src.grounding.pipeline import apply_grounding
from src.grounding.citation import extract_cited_pages, is_unanswerable
from src.ingest.pipeline import ingest_document
from src.refusal.decision import apply_refusal, build_refusal_response
from src.refusal.score_gate import max_retrieval_score
from src.refusal.calibration import score_signature, validate_artifact
from src.retrieval.hybrid_fusion import fuse_retrieval
from src.retrieval.index_store import DocumentIndex
from src.utils.config import doc_id_from_pdf
from src.utils.runtime import choose_backends
from src.utils.text import content_tokens, evidence_window
from src.retrieval.hybrid_fusion import absolute_score

logger = logging.getLogger(__name__)


def _release_cuda_memory():
    """Release optional CUDA allocations without making import-time torch mandatory."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, OSError):
        # Importing the lightweight/test-only parts of this module must not
        # depend on a working PyTorch CUDA installation.
        return


def _repair_missing_citation(raw_answer, retrieved_hits, grounding_cfg):
    """Opt-in repair only after a textual support check; off for experiments."""
    if not grounding_cfg.get("repair_missing_citations", False):
        return raw_answer, False
    if not raw_answer or is_unanswerable(raw_answer) or extract_cited_pages(raw_answer):
        return raw_answer, False

    answer = raw_answer.strip()
    # A response such as "[3]" contains no claim to ground.
    if re.fullmatch(r"\[\s*(?:p\.)?\s*\d+\s*\]", answer, flags=re.IGNORECASE):
        return raw_answer, False

    # Recover a citation only after checking evidence; never attach top-1 blindly.
    from src.grounding.claims import check_text_support
    if not grounding_cfg.get("verify", True):
        return raw_answer, False
    for hit in retrieved_hits:
        evidence = hit.get("text") or hit.get("text_snippet", "")
        if not check_text_support(answer, evidence, min_overlap=0.8)["supported"]:
            continue
        page = hit.get("page_number")
        if page is None and hit.get("page_index") is not None:
            page = hit["page_index"] + 1
        if page is not None:
            return "{0} [p.{1}]".format(answer, page), True
    return raw_answer, False


def _format_hybrid_pages(fused_hits):
    return [
        {
            **h,
            "page_index": h["page_index"],
            "page_number": h["page_index"] + 1,
            "page_path": h.get("page_path"),
            "fused_score": round(h["fused_score"], 4),
            "visual_score": round(h.get("visual_score", h.get("score", 0.0)), 4),
            "text_score": round(h.get("text_score", 0.0), 4),
            "visual_weight": h.get("visual_weight"),
            "text_weight": h.get("text_weight"),
            "text_snippet": h.get("text_snippet", ""),
        }
        for h in fused_hits
    ]


def _make_visual_retriever(visual_backend, cfg):
    if visual_backend == "colpali":
        from src.retrieval.colpali_retriever import ColPaliRetriever

        return ColPaliRetriever(model_name=cfg["retrieval"]["model"], device=cfg.get("runtime", {}).get("device"),
                                revision=cfg["retrieval"].get("model_revision"))
    if visual_backend == "clip":
        from src.retrieval.clip_retriever import CLIPRetriever

        return CLIPRetriever(model_name=cfg["retrieval"].get("clip_model", "openai/clip-vit-base-patch32"),
                             device=cfg.get("runtime", {}).get("device"), revision=cfg["retrieval"].get("clip_revision"))
    return None


def _retrieve_hybrid(question, page_paths, bm25, visual_embeddings, cfg, visual_backend):
    top_k = cfg["retrieval"]["top_k"]
    fusion_cfg = cfg["retrieval"].get("fusion", {})
    candidate_k = max(top_k, cfg["retrieval"].get("candidate_k", 20))

    visual_hits = []
    if visual_backend not in (None, "none"):
        retriever = _make_visual_retriever(visual_backend, cfg)
        visual_hits = retriever.retrieve(
            question, page_paths, top_k=candidate_k, cached_embeddings=visual_embeddings
        )
        del retriever
        _release_cuda_memory()

    text_hits = bm25.retrieve(question, top_k=candidate_k)
    text_densities = bm25.get_text_densities()

    query_tokens = set(content_tokens(question))
    page_features = {p["page_index"]: {**p,
        "query_coverage": len(query_tokens & set(content_tokens(p["text"]))) / max(len(query_tokens), 1)
    } for p in bm25.pages}
    fused_hits = fuse_retrieval(
        visual_hits,
        text_hits,
        page_text_densities=text_densities,
        top_k=top_k,
        min_text_weight=fusion_cfg.get("min_text_weight", 0.25),
        max_text_weight=fusion_cfg.get("max_text_weight", 0.75),
        mode=fusion_cfg.get("mode", "reliability"),
        fixed_text_weight=fusion_cfg.get("fixed_text_weight", 0.5),
        visual_backend=visual_backend,
        text_scale=fusion_cfg.get("text_scale", 8.0),
        visual_scale=fusion_cfg.get("visual_scale", 20.0),
        rrf_k=fusion_cfg.get("rrf_k", 60),
        page_features=page_features,
        use_query_coverage=fusion_cfg.get("use_query_coverage", True),
        use_text_quality=fusion_cfg.get("use_text_quality", True),
    )
    # Attach snippets for lexical verify / extractive gen
    snippet_by_page = {h["page_index"]: h.get("text_snippet", "") for h in text_hits}
    # Also fill from full BM25 pages if missing
    for page in bm25.pages:
        snippet_by_page.setdefault(page["page_index"], evidence_window(page["text"], question))
    for h in fused_hits:
        if not h.get("text_snippet"):
            h["text_snippet"] = snippet_by_page.get(h["page_index"], "")
    return visual_hits, text_hits, fused_hits


def _retrieve_visual(question, page_paths, visual_embeddings, cfg, visual_backend):
    top_k = cfg["retrieval"]["top_k"]
    retriever = _make_visual_retriever(visual_backend, cfg)
    if retriever is None:
        return [], [], []
    visual_hits = retriever.retrieve(
        question, page_paths, top_k=top_k, cached_embeddings=visual_embeddings
    )
    del retriever
    _release_cuda_memory()

    hits = [
        {
            "page_index": h["page_index"],
            "page_number": h["page_index"] + 1,
            "page_path": h["page_path"],
            "score": h["score"],
            "fused_score": h["score"],
            "evidence_confidence": absolute_score(h["score"], "visual", visual_backend,
                                                  visual_scale=cfg["retrieval"].get("fusion", {}).get("visual_scale", 20.0)),
        }
        for h in visual_hits
    ]
    return visual_hits, [], hits


def _retrieve_ocr(question, bm25, cfg):
    top_k = cfg["retrieval"]["top_k"]
    text_hits = bm25.retrieve(question, top_k=top_k)
    hits = [
        {
            **h,
            "page_index": h["page_index"],
            "page_number": h["page_index"] + 1,
            "page_path": h["page_path"],
            "score": h["score"],
            "fused_score": h["score"],
            "text_snippet": h.get("text_snippet", ""),
            "evidence_confidence": absolute_score(h["score"], "text", text_scale=cfg["retrieval"].get("fusion", {}).get("text_scale", 8.0)),
        }
        for h in text_hits
    ]
    return [], text_hits, hits


def _enrich_snippets(retrieved_hits, bm25, question=""):
    if bm25 is None:
        return retrieved_hits
    text_by_idx = {p["page_index"]: p["text"] for p in bm25.pages}
    for h in retrieved_hits:
        h["text"] = text_by_idx.get(h["page_index"], h.get("text", ""))
        if not h.get("text_snippet"):
            text = text_by_idx.get(h["page_index"], "")
            h["text_snippet"] = evidence_window(text, question)
    return retrieved_hits


def run_rag(pdf_path, question, cfg, method="full", force_rebuild=False):
    """Run end-to-end RAG with configurable method and features.

    Methods:
        full / hybrid — adaptive visual+text fusion
        visual — visual channel only (ColPali or CLIP)
        text / ocr — native text / actual OCR, BM25 retrieval and text generation
        demo — forced CLIP/none + extractive (alias for laptop path)
    """
    started = perf_counter()
    timings = {}
    cfg = copy.deepcopy(cfg)
    if method == "ocr":
        cfg.setdefault("pdf", {})["text_mode"] = "ocr"
    elif method == "text":
        cfg.setdefault("pdf", {})["text_mode"] = "native"
    pdf_path = str(Path(pdf_path).resolve())
    doc_index = DocumentIndex(doc_id_from_pdf(pdf_path))
    refusal_cfg = dict(cfg.get("refusal", {}) or {})
    grounding_cfg = dict(cfg.get("grounding", {}) or {})

    if method == "demo":
        cfg = dict(cfg)
        runtime = dict(cfg.get("runtime", {}) or {})
        runtime["demo_mode"] = True
        runtime["visual_backend"] = runtime.get("visual_backend") or "clip"
        runtime["generation_backend"] = "extractive"
        cfg["runtime"] = runtime
        method = "full"

    backends = choose_backends(cfg)
    visual_backend = backends["visual_backend"]
    generation_backend = backends["generation_backend"]

    # OCR baseline never needs visual; visual baseline needs a visual backend.
    if method in ("text", "ocr"):
        visual_backend = "none"
    if method == "visual" and visual_backend == "none":
        visual_backend = "clip"

    # Grounding is meaningful for answer-producing retrieval methods.  Refusal
    # is an independent feature switch: restricting it to ``method == full``
    # made the no_grounding/no_refusal ablations identical because both used
    # the ``hybrid`` method name.
    grounding_enabled = grounding_cfg.get("enabled", False)
    refusal_enabled = refusal_cfg.get("enabled", False)

    # Explicit config thresholds always take precedence over defaults.
    if generation_backend == "extractive":
        grounding_cfg.setdefault("verify_mode", "lexical")
        # Absolute CLIP/BM25 fused scores on demo docs are usually mid-range
        if cfg.get("runtime", {}).get("demo_mode"):
            refusal_cfg.setdefault("score_threshold", refusal_cfg.get("demo_score_threshold", 0.05))

    signature = score_signature(cfg, method, visual_backend)
    if refusal_cfg.get("calibration_path"):
        artifact = json.loads(Path(refusal_cfg["calibration_path"]).read_text(encoding="utf-8"))
        refusal_cfg["score_threshold"] = validate_artifact(artifact, signature)
    backends = {**backends, "visual_backend": visual_backend}
    if backends["device"] == "cuda":
        import torch
        torch.cuda.reset_peak_memory_stats()
    def finish(result):
        result["latency_seconds"] = round(perf_counter()-started, 6)
        result["evidence_confidence"] = max_retrieval_score(retrieved_hits)
        result["score_signature"] = signature
        result["effective_text_mode"] = cfg.get("pdf", {}).get("text_mode", "native")
        result["indexed_page_count"] = len(page_paths)
        result["timings"] = timings
        if backends["device"] == "cuda":
            result["peak_cuda_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 2)
        else:
            result["peak_cuda_allocated_mb"] = None
        return result

    page_paths, bm25, visual_embeddings, _ = ingest_document(
        pdf_path,
        cfg,
        doc_index=doc_index,
        force_rebuild=force_rebuild,
        visual_backend=visual_backend,
    )
    indexed_at = perf_counter()
    timings["ingest_seconds"] = round(indexed_at-started, 6)

    if method in ("full", "hybrid"):
        visual_hits, text_hits, retrieved_hits = _retrieve_hybrid(
            question, page_paths, bm25, visual_embeddings, cfg, visual_backend
        )
        method_name = "hybrid_rag_grounded" if grounding_enabled else "hybrid_rag"
        if method == "full" and refusal_enabled:
            method_name = "full_rag"
        method_name = "{0}_{1}_{2}".format(method_name, visual_backend, generation_backend)
    elif method == "visual":
        visual_hits, text_hits, retrieved_hits = _retrieve_visual(
            question, page_paths, visual_embeddings, cfg, visual_backend
        )
        method_name = "visual_rag_{0}".format(visual_backend)
    elif method in ("text", "ocr"):
        visual_hits, text_hits, retrieved_hits = _retrieve_ocr(question, bm25, cfg)
        method_name = f"{method}_rag"
    else:
        raise ValueError("Unknown method: {0}".format(method))

    retrieved_hits = _enrich_snippets(retrieved_hits, bm25, question)
    retrieved_at = perf_counter()
    timings["retrieval_seconds"] = round(retrieved_at-indexed_at, 6)

    # Pre-generation retrieval refusal
    if refusal_enabled:
        pre_refusal = apply_refusal(
            retrieved_hits, raw_answer=None, grounding_result=None, refusal_cfg=refusal_cfg
        )
        if pre_refusal.get("refused") and pre_refusal.get("refusal_reason") == "retrieval_miss":
            result = build_refusal_response(
                "retrieval_miss",
                retrieval_score=pre_refusal["max_retrieval_score"],
            )
            result.update(
                {
                    "question": question,
                    "method": method_name,
                    "retrieved_pages": _format_hybrid_pages(retrieved_hits)
                    if method in ("full", "hybrid")
                    else retrieved_hits,
                    "refusal": pre_refusal,
                    "backends": backends,
                }
            )
            return finish(result)

    top_paths = [h["page_path"] for h in retrieved_hits if h.get("page_path")]
    page_numbers = [h.get("page_number") or (h["page_index"] + 1) for h in retrieved_hits if h.get("page_path")]
    evidence_texts = ([evidence_window(h.get("text", ""), question,
        cfg.get("generation", {}).get("max_context_chars_per_page", 800)) for h in retrieved_hits if h.get("page_path")]
        if method in ("full", "hybrid") and cfg.get("generation", {}).get("use_text_evidence", True) else None)

    model = processor = None
    grounding = None
    answer_with_citations = None
    citation_repaired = False

    if generation_backend == "extractive":
        raw_answer = answer_extractive_with_citations(question, retrieved_hits)
        if grounding_enabled:
            answer_with_citations, citation_repaired = _repair_missing_citation(
                raw_answer, retrieved_hits, grounding_cfg
            )
            grounding = apply_grounding(
                None, None, answer_with_citations, retrieved_hits, grounding_cfg, question=question
            )
            answer = grounding.get("answer_text") or grounding.get("answer")
        else:
            answer = raw_answer
    else:
        load_in_4bit = bool(cfg.get("generation", {}).get("load_in_4bit", False))
        adapter_path = cfg.get("generation", {}).get("adapter_path")
        model, processor = load_vlm(cfg["generation"]["model"], load_in_4bit=load_in_4bit,
                                    device=backends["device"], revision=cfg["generation"].get("revision"),
                                    adapter_path=adapter_path)

        if method in ("text", "ocr"):
            context_parts = []
            for h in retrieved_hits:
                snippet = h.get("text_snippet", "")
                context_parts.append("[Page {0}]\n{1}".format(h["page_number"], snippet))
            text_context = "\n\n".join(context_parts) if context_parts else "(no relevant text found)"
            raw_answer = answer_with_text_context(
                model,
                processor,
                question,
                text_context,
                max_new_tokens=cfg["generation"]["max_new_tokens"], grounded=grounding_enabled,
                retry_feedback=cfg["generation"].get("retry_feedback"),
            )
            if grounding_enabled:
                grounding = apply_grounding(model, processor, raw_answer, retrieved_hits,
                                            grounding_cfg, question=question)
                answer = grounding.get("answer_text") or raw_answer
            else:
                answer = raw_answer
        elif grounding_enabled and top_paths:
            raw_answer = answer_with_grounding(
                model,
                processor,
                question,
                top_paths,
                page_numbers=page_numbers,
                max_new_tokens=cfg["generation"]["max_new_tokens"],
                cite_format=grounding_cfg.get("cite_format", "[p.{page}]"),
                retry_feedback=cfg["generation"].get("retry_feedback"),
                evidence_texts=evidence_texts,
                prompt_language=grounding_cfg.get("prompt_language", "zh"),
            )
            answer_with_citations, citation_repaired = _repair_missing_citation(
                raw_answer, retrieved_hits, grounding_cfg
            )
            grounding = apply_grounding(
                model, processor, answer_with_citations, retrieved_hits, grounding_cfg, question=question
            )
            answer = grounding.get("answer_text") or grounding.get("answer")
        elif top_paths:
            raw_answer = answer_with_vlm(
                model,
                processor,
                question,
                top_paths,
                max_new_tokens=cfg["generation"]["max_new_tokens"],
                evidence_texts=evidence_texts, page_numbers=page_numbers,
            )
            answer = raw_answer
        else:
            raw_answer = "UNANSWERABLE"
            answer = raw_answer

    result = {
        "question": question,
        "answer": answer,
        "raw_answer": raw_answer,
        "answer_with_citations": answer_with_citations or raw_answer,
        "method": method_name,
        "retrieved_pages": _format_hybrid_pages(retrieved_hits)
        if method in ("full", "hybrid")
        else retrieved_hits,
        "answerable": not is_unanswerable(raw_answer),
        "refused": is_unanswerable(raw_answer),
        "model_abstained": is_unanswerable(raw_answer),
        "policy_refused": False,
        "backends": {
            "visual_backend": visual_backend,
            "generation_backend": generation_backend,
            "device": backends["device"],
        },
    }
    timings["generation_and_verification_seconds"] = round(perf_counter()-retrieved_at, 6)

    if method in ("full", "hybrid"):
        result["visual_only_pages"] = [
            {"page_index": h["page_index"], "score": h["score"]} for h in visual_hits
        ]
        result["text_only_pages"] = [
            {"page_index": h["page_index"], "score": h["score"]} for h in text_hits
        ]

    if grounding is not None:
        result["grounding"] = {
            "answerable": grounding["answerable"],
            "citations": grounding["citations"],
            "citations_valid": grounding["citations_valid"],
            "invalid_citations": grounding["invalid_citations"],
            "grounding_verified": grounding["grounding_verified"],
            "confidence": grounding["confidence"],
            "verification_results": grounding["verification_results"],
            "verification_reason": grounding.get("verification_reason"),
            "failed_citations": grounding.get("failed_citations", []),
            "feedback": grounding.get("feedback"),
            "verification_backend": grounding.get("verification_backend"),
            "claim_results": grounding.get("claim_results", []),
            "claim_coverage": grounding.get("claim_coverage"),
            "citation_repaired": citation_repaired,
        }

    if refusal_enabled:
        post_refusal = apply_refusal(
            retrieved_hits,
            raw_answer=raw_answer,
            grounding_result=grounding,
            refusal_cfg=refusal_cfg,
        )
        result["refusal"] = post_refusal
        if post_refusal.get("refused"):
            result["answerable"] = False
            result["refused"] = True
            result["policy_refused"] = True
            result["answer"] = None
            result["refusal_reason"] = post_refusal["refusal_reason"]
            result["message"] = post_refusal.get("message")

    if result["model_abstained"] and not result.get("refusal_reason"):
        result["refusal_reason"] = "model_refusal"
        result["answer"] = None
    # Do not retain VLM allocations between independent requests on a 6GB GPU.
    if model is not None:
        del model, processor
        _release_cuda_memory()
    return finish(result)
