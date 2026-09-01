"""Minimal visual RAG baseline: retrieve pages then answer with VLM."""

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation.vlm_generator import answer_with_vlm, load_vlm
from src.ingest.pipeline import ingest_document
from src.retrieval.colpali_retriever import ColPaliRetriever
from src.retrieval.index_store import DocumentIndex
from src.utils.config import doc_id_from_pdf, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=str, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    doc_id = doc_id_from_pdf(args.pdf)
    doc_index = DocumentIndex(doc_id)

    page_paths, _, visual_embeddings, _ = ingest_document(
        args.pdf, cfg, doc_index=doc_index, force_rebuild=args.force_rebuild
    )

    retriever = ColPaliRetriever(model_name=cfg["retrieval"]["model"])
    hits = retriever.retrieve(
        args.question, page_paths,
        top_k=cfg["retrieval"]["top_k"],
        cached_embeddings=visual_embeddings,
    )
    top_paths = [h["page_path"] for h in hits]

    del retriever
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model, processor = load_vlm(cfg["generation"]["model"])
    answer = answer_with_vlm(
        model, processor, args.question, top_paths,
        max_new_tokens=cfg["generation"]["max_new_tokens"],
    )

    result = {
        "question": args.question,
        "answer": answer,
        "method": "visual_rag",
        "retrieved_pages": [
            {"page_index": h["page_index"], "page_number": h["page_index"] + 1, "score": h["score"]}
            for h in hits
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
