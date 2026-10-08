"""Document ingestion: render pages, extract text, build/load persistent index."""

from __future__ import annotations

import logging
from pathlib import Path

from src.ingest.pdf_loader import pdf_to_page_images, pdf_to_text_chunks
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.index_store import DocumentIndex

logger = logging.getLogger(__name__)


def _load_visual_embeddings(page_paths, cfg, doc_index, visual_backend, fresh):
    """Build or load cached visual embeddings for the selected backend."""
    cache_visual = cfg["retrieval"].get("cache_visual_embeddings", True)
    if visual_backend in (None, "none"):
        return None

    # Each backend gets an independent signature so rebuilding the text index
    # cannot accidentally bless stale vectors from another backend.
    manifest = doc_index.load_manifest() or {}
    signature = manifest.get("fingerprint", "legacy")
    emb_name = f"visual_{visual_backend}_{signature}.pt"

    if cache_visual and fresh:
        path = doc_index.root / emb_name
        if path.exists():
            import torch
            return torch.load(path, weights_only=True, map_location="cpu")

    if visual_backend == "colpali":
        from src.retrieval.colpali_retriever import ColPaliRetriever

        retriever = ColPaliRetriever(model_name=cfg["retrieval"]["model"],
                                    device=cfg.get("runtime", {}).get("device"),
                                    revision=cfg["retrieval"].get("model_revision"))
        embeddings = retriever.embed_pages(page_paths)
        if cache_visual:
            import torch
            torch.save(embeddings.cpu(), doc_index.root / emb_name)
        del retriever
        return embeddings

    if visual_backend == "clip":
        from src.retrieval.clip_retriever import CLIPRetriever

        model_name = cfg["retrieval"].get("clip_model", "openai/clip-vit-base-patch32")
        retriever = CLIPRetriever(model_name=model_name, device=cfg.get("runtime", {}).get("device"),
                                 revision=cfg["retrieval"].get("clip_revision"))
        embeddings = retriever.embed_pages(page_paths)
        if cache_visual:
            import torch

            torch.save(embeddings, doc_index.root / emb_name)
        del retriever
        return embeddings

    logger.warning("Unknown visual backend %s; skipping visual embeddings", visual_backend)
    return None


def ingest_document(pdf_path, cfg, doc_index=None, force_rebuild=False, visual_backend="colpali"):
    """Render PDF pages, build/load BM25 index and optional visual embeddings.

    Returns:
        page_paths: list[str]
        bm25: BM25Retriever (indexed)
        visual_embeddings: torch.Tensor | None
        text_densities: dict[int, int]
    """
    pdf_path = str(Path(pdf_path).resolve())
    dpi = cfg["pdf"]["dpi"]
    max_pages = cfg["pdf"].get("max_pages_per_doc")

    if doc_index is None:
        from src.utils.config import doc_id_from_pdf
        doc_index = DocumentIndex(doc_id_from_pdf(pdf_path))

    signature = doc_index.fingerprint(pdf_path, cfg)
    # Different OCR/model/render profiles must never overwrite each other's
    # pages or manifests (including concurrent CPU and GPU experiments).
    doc_index = DocumentIndex(signature[:24], cache_dir=doc_index.root)
    page_dir = doc_index.root / "pages"
    manifest = doc_index.load_manifest() or {}
    fresh = manifest.get("ready") is True and manifest.get("fingerprint") == signature and not force_rebuild

    if fresh and manifest.get("page_paths") and all(Path(p).exists() for p in manifest["page_paths"]):
        manifest = doc_index.load_manifest()
        page_paths = manifest["page_paths"]
    else:
        page_paths = pdf_to_page_images(pdf_path, str(page_dir), dpi=dpi, max_pages=max_pages)
        doc_index.save_manifest(
            {
                "pdf_path": pdf_path,
                "pdf_mtime": Path(pdf_path).stat().st_mtime,
                "dpi": dpi,
                "max_pages": max_pages,
                "page_paths": page_paths,
                "num_pages": len(page_paths),
                "visual_backend": visual_backend,
                "fingerprint": signature,
                "ready": False,
            }
        )
        fresh = False

    bm25 = BM25Retriever()
    if fresh:
        bm25_state = doc_index.load_bm25()
        if bm25_state:
            bm25.load_state(bm25_state)
        else:
            fresh = False

    if not fresh or bm25.bm25 is None:
        text_chunks = pdf_to_text_chunks(pdf_path, max_pages=max_pages, **{
            key: cfg["pdf"][key] for key in ("text_mode", "ocr_language", "ocr_dpi", "tessdata", "min_native_chars")
            if key in cfg["pdf"]
        })
        bm25.build_index(page_paths, text_chunks)
        doc_index.save_bm25(bm25.dump_state())

    visual_embeddings = _load_visual_embeddings(
        page_paths, cfg, doc_index, visual_backend, fresh
    )
    manifest = doc_index.load_manifest()
    manifest["ready"] = True
    doc_index.save_manifest(manifest)
    return page_paths, bm25, visual_embeddings, bm25.get_text_densities()
