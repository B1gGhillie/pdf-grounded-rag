"""Unified RAG pipeline: retrieval, refusal, generation, grounding."""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import torch

from src.generation.extractive_generator import answer_extractive_with_citations
from src.generation.vlm_generator import (
    answer_with_grounding,
    answer_with_text_context,
    answer_with_vlm,
    load_vlm,
)
from src.grounding.pipeline import apply_grounding
from src.ingest.pipeline import ingest_document
from src.refusal.decision import apply_refusal, build_refusal_response
from src.retrieval.hybrid_fusion import fuse_retrieval
from src.retrieval.index_store import DocumentIndex
from src.utils.config import doc_id_from_pdf
from src.utils.runtime import choose_backends

logger = logging.getLogger(__name__)


def _format_hybrid_pages(fused_hits):
    return [
        {
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

        return ColPaliRetriever(model_name=cfg["retrieval"]["model"])
    if visual_backend == "clip":
        from src.retrieval.clip_retriever import CLIPRetriever

        return CLIPRetriever(model_name=cfg["retrieval"].get("clip_model", "openai/clip-vit-base-patch32"))
    return None


def _retrieve_hybrid(question, page_paths, bm25, visual_embeddings, cfg, visual_backend):
    top_k = cfg["retrieval"]["top_k"]
    fusion_cfg = cfg["retrieval"].get("fusion", {})

    visual_hits = []
    if visual_backend not in (None, "none"):
        retriever = _make_visual_retriever(visual_backend, cfg)
        visual_hits = retriever.retrieve(
            question, page_paths, top_k=top_k, cached_embeddings=visual_embeddings
        )
        del retriever
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    text_hits = bm25.retrieve(question, top_k=top_k)
    text_densities = bm25.get_text_densities()

    if not visual_hits:
        # Text-only fusion path: treat BM25 as the sole channel.
        fused_hits = []
        max_raw = max((h["score"] for h in text_hits), default=1.0) or 1.0
        for h in text_hits:
            fused_hits.append(
                {
                    "page_index": h["page_index"],
                    "page_path": h["page_path"],
                    "fused_score": h["score"] / max_raw,
                    "visual_score": 0.0,
                    "text_score": h["score"] / max_raw,
                    "visual_weight": 0.0,
                    "text_weight": 1.0,
                    "text_snippet": h.get("text_snippet", ""),
                }
            )
        return visual_hits, text_hits, fused_hits[:top_k]

    fused_hits = fuse_retrieval(
        visual_hits,
        text_hits,
        page_text_densities=text_densities,
        top_k=top_k,
        min_text_weight=fusion_cfg.get("min_text_weight", 0.25),
        max_text_weight=fusion_cfg.get("max_text_weight", 0.75),
    )
    # Attach snippets for lexical verify / extractive gen
    snippet_by_page = {h["page_index"]: h.get("text_snippet", "") for h in text_hits}
    # Also fill from full BM25 pages if missing
    for page in bm25.pages:
        snippet_by_page.setdefault(page["page_index"], page["text"][:300])
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
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    hits = [
        {
            "page_index": h["page_index"],
            "page_number": h["page_index"] + 1,
            "page_path": h["page_path"],
            "score": h["score"],
            "fused_score": h["score"],
        }
        for h in visual_hits
    ]
    return visual_hits, [], hits


def _retrieve_ocr(question, bm25, cfg):
    top_k = cfg["retrieval"]["top_k"]
    text_hits = bm25.retrieve(question, top_k=top_k)
    hits = [
        {
            "page_index": h["page_index"],
            "page_number": h["page_index"] + 1,
            "page_path": h["page_path"],
            "score": h["score"],
            "fused_score": h["score"],
            "text_snippet": h.get("text_snippet", ""),
        }
        for h in text_hits
    ]
    return [], text_hits, hits


def _enrich_snippets(retrieved_hits, bm25):
    if bm25 is None:
        return retrieved_hits
    text_by_idx = {p["page_index"]: p["text"] for p in bm25.pages}
    for h in retrieved_hits:
        if not h.get("text_snippet"):
            text = text_by_idx.get(h["page_index"], "")
            h["text_snippet"] = text[:300]
    return retrieved_hits


def run_rag(pdf_path, question, cfg, method="full", force_rebuild=False):
    """Run end-to-end RAG with configurable method and features.

    Methods:
        full / hybrid — adaptive visual+text fusion
        visual — visual channel only (ColPali or CLIP)
        ocr — BM25 text only
        demo — forced CLIP/none + extractive (alias for laptop path)
    """
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
    if method == "ocr":
        visual_backend = "none"
    if method == "visual" and visual_backend == "none":
        visual_backend = "clip"

    grounding_enabled = grounding_cfg.get("enabled", False) and method in ("full", "hybrid")
    refusal_enabled = refusal_cfg.get("enabled", False) and method == "full"

    # Extractive + lexical verify defaults for laptop path
    if generation_backend == "extractive":
        grounding_cfg.setdefault("verify_mode", "lexical")
        # Absolute CLIP/BM25 fused scores on demo docs are usually mid-range
        if cfg.get("runtime", {}).get("demo_mode") or backends["probe"].get("free_ram_gb", 99) < 12:
            refusal_cfg.setdefault("score_threshold", refusal_cfg.get("demo_score_threshold", 0.05))

    page_paths, bm25, visual_embeddings, _ = ingest_document(
        pdf_path,
        cfg,
        doc_index=doc_index,
        force_rebuild=force_rebuild,
        visual_backend=visual_backend,
    )

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
    elif method == "ocr":
        visual_hits, text_hits, retrieved_hits = _retrieve_ocr(question, bm25, cfg)
        method_name = "ocr_rag"
    else:
        raise ValueError("Unknown method: {0}".format(method))

    retrieved_hits = _enrich_snippets(retrieved_hits, bm25)

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
            return result

    top_paths = [h["page_path"] for h in retrieved_hits if h.get("page_path")]
    page_numbers = [h.get("page_number") or (h["page_index"] + 1) for h in retrieved_hits if h.get("page_path")]

    model = processor = None
    grounding = None

    if generation_backend == "extractive":
        raw_answer = answer_extractive_with_citations(question, retrieved_hits)
        if grounding_enabled:
            grounding = apply_grounding(None, None, raw_answer, retrieved_hits, grounding_cfg)
            answer = grounding.get("answer_text") or grounding.get("answer")
        else:
            answer = raw_answer
    else:
        load_in_4bit = bool(cfg.get("generation", {}).get("load_in_4bit", False))
        model, processor = load_vlm(cfg["generation"]["model"], load_in_4bit=load_in_4bit)

        if method == "ocr":
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
                max_new_tokens=cfg["generation"]["max_new_tokens"],
            )
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
            )
            grounding = apply_grounding(model, processor, raw_answer, retrieved_hits, grounding_cfg)
            answer = grounding.get("answer_text") or grounding.get("answer")
        elif top_paths:
            raw_answer = answer_with_vlm(
                model,
                processor,
                question,
                top_paths,
                max_new_tokens=cfg["generation"]["max_new_tokens"],
            )
            answer = raw_answer
        else:
            raw_answer = "UNANSWERABLE"
            answer = raw_answer

    result = {
        "question": question,
        "answer": answer,
        "raw_answer": raw_answer,
        "method": method_name,
        "retrieved_pages": _format_hybrid_pages(retrieved_hits)
        if method in ("full", "hybrid")
        else retrieved_hits,
        "answerable": True,
        "refused": False,
        "backends": {
            "visual_backend": visual_backend,
            "generation_backend": generation_backend,
            "device": backends["device"],
        },
    }

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
            "verification_reason": grounding["verification_reason"],
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
            result["answer"] = None
            result["refusal_reason"] = post_refusal["refusal_reason"]
            result["message"] = post_refusal.get("message")

    return result
