"""OCR-RAG baseline: BM25 text retrieval -> text-context generation."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation.vlm_generator import answer_with_text_context, load_vlm
from src.ingest.pipeline import ingest_document
from src.retrieval.index_store import DocumentIndex
from src.utils.config import doc_id_from_pdf, load_config


def run_ocr_rag(pdf_path, question, cfg, force_rebuild=False):
    doc_id = doc_id_from_pdf(pdf_path)
    doc_index = DocumentIndex(doc_id)

    page_paths, bm25, _, _ = ingest_document(
        pdf_path, cfg, doc_index=doc_index, force_rebuild=force_rebuild
    )
    top_k = cfg["retrieval"]["top_k"]
    text_hits = bm25.retrieve(question, top_k=top_k)

    context_parts = []
    for h in text_hits:
        page_num = h["page_index"] + 1
        snippet = h.get("text_snippet", "")
        context_parts.append("[Page {0}]\n{1}".format(page_num, snippet))
    text_context = "\n\n".join(context_parts) if context_parts else "(no relevant text found)"

    model, processor = load_vlm(cfg["generation"]["model"])
    answer = answer_with_text_context(
        model, processor, question, text_context,
        max_new_tokens=cfg["generation"]["max_new_tokens"],
    )

    return {
        "question": question,
        "answer": answer,
        "method": "ocr_rag",
        "retrieved_pages": [
            {
                "page_index": h["page_index"],
                "page_number": h["page_index"] + 1,
                "score": round(h["score"], 4),
            }
            for h in text_hits
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="OCR-RAG baseline: BM25 + text generation")
    parser.add_argument("--pdf", type=str, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = run_ocr_rag(args.pdf, args.question, cfg, force_rebuild=args.force_rebuild)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
