"""CPU demo RAG: BM25 retrieval + extractive cited answers + grounding + refusal.

Use this when ColPali / Qwen2-VL are not available (broken CUDA env / no GPU RAM).
It still exercises page-level citations, Cite validation, and multi-signal refusal.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation.extractive_generator import answer_extractive_with_citations
from src.grounding.pipeline import apply_grounding
from src.ingest.pdf_loader import pdf_to_page_images, pdf_to_text_chunks
from src.refusal.decision import apply_refusal, build_refusal_response
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.index_store import DocumentIndex
from src.utils.config import doc_id_from_pdf, load_config


def ingest_text_only(pdf_path, cfg, force_rebuild=False):
    pdf_path = str(Path(pdf_path).resolve())
    dpi = cfg["pdf"]["dpi"]
    max_pages = cfg["pdf"].get("max_pages_per_doc", 50)
    doc_index = DocumentIndex(doc_id_from_pdf(pdf_path))
    page_dir = doc_index.root / "pages"

    fresh = doc_index.is_fresh(pdf_path, dpi, max_pages) and not force_rebuild
    if fresh and (page_dir / "page_0001.png").exists():
        page_paths = doc_index.load_manifest()["page_paths"]
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
            }
        )

    bm25 = BM25Retriever()
    if fresh:
        state = doc_index.load_bm25()
        if state:
            bm25.load_state(state)
        else:
            fresh = False
    if not fresh or bm25.bm25 is None:
        chunks = pdf_to_text_chunks(pdf_path, max_pages=max_pages)
        bm25.build_index(page_paths, chunks)
        doc_index.save_bm25(bm25.dump_state())
    return page_paths, bm25


def run_demo(pdf_path, question, cfg):
    refusal_cfg = dict(cfg.get("refusal", {}))
    grounding_cfg = dict(cfg.get("grounding", {}))
    # Demo: no VLM verifier available
    grounding_cfg["verify"] = False
    refusal_cfg.setdefault("use_citation_gate", True)
    # Absolute BM25 scores are not in [0,1]; disable score gate or use low threshold after norm
    # We normalize BM25 hits into fused_score via simple max-norm below.
    refusal_cfg["score_threshold"] = refusal_cfg.get("demo_score_threshold", 0.05)

    _, bm25 = ingest_text_only(pdf_path, cfg)
    top_k = cfg["retrieval"]["top_k"]
    text_hits = bm25.retrieve(question, top_k=top_k)

    if not text_hits:
        result = build_refusal_response("retrieval_miss", retrieval_score=0.0)
        result.update({"question": question, "method": "demo_extractive_rag", "retrieved_pages": []})
        return result

    max_raw = max(h["score"] for h in text_hits) or 1.0
    retrieved_hits = []
    for h in text_hits:
        retrieved_hits.append(
            {
                "page_index": h["page_index"],
                "page_number": h["page_index"] + 1,
                "page_path": h["page_path"],
                "score": h["score"],
                "fused_score": h["score"] / max_raw,
                "text_snippet": h.get("text_snippet", ""),
            }
        )

    pre = apply_refusal(retrieved_hits, raw_answer=None, grounding_result=None, refusal_cfg=refusal_cfg)
    if pre.get("refused") and pre.get("refusal_reason") == "retrieval_miss":
        result = build_refusal_response("retrieval_miss", retrieval_score=pre["max_retrieval_score"])
        result.update(
            {
                "question": question,
                "method": "demo_extractive_rag",
                "retrieved_pages": retrieved_hits,
                "refusal": pre,
            }
        )
        return result

    raw_answer = answer_extractive_with_citations(question, retrieved_hits)
    grounding = apply_grounding(None, None, raw_answer, retrieved_hits, grounding_cfg)
    answer = grounding.get("answer_text") or grounding.get("answer")

    result = {
        "question": question,
        "answer": answer,
        "raw_answer": raw_answer,
        "method": "demo_extractive_rag",
        "retrieved_pages": [
            {
                "page_number": h["page_number"],
                "fused_score": round(h["fused_score"], 4),
                "text_snippet": h["text_snippet"][:180],
            }
            for h in retrieved_hits
        ],
        "grounding": {
            "answerable": grounding["answerable"],
            "citations": grounding["citations"],
            "citations_valid": grounding["citations_valid"],
            "invalid_citations": grounding["invalid_citations"],
            "grounding_verified": grounding["grounding_verified"],
            "confidence": grounding["confidence"],
            "verification_reason": grounding["verification_reason"],
        },
        "answerable": True,
        "refused": False,
        "note": "CPU demo: BM25 + extractive cite (no ColPali/VLM). Grounding verify disabled.",
    }

    post = apply_refusal(
        retrieved_hits,
        raw_answer=raw_answer,
        grounding_result=grounding,
        refusal_cfg=refusal_cfg,
    )
    result["refusal"] = post
    if post.get("refused"):
        result["answerable"] = False
        result["refused"] = True
        result["answer"] = None
        result["refusal_reason"] = post["refusal_reason"]
        result["message"] = post.get("message")
    return result


DEFAULT_QUESTIONS = [
    "What is the main contribution of this document?",
    "What accuracy did hybrid RAG achieve?",
    "Which model is used for multimodal answer generation?",
    "Who is the CEO of OpenAI?",
    "What training hyperparameters were used for ColPali in this report?",
]


def main():
    parser = argparse.ArgumentParser(description="CPU demo RAG on a PDF")
    parser.add_argument("--pdf", type=str, default="data/pdfs/sample_doc.pdf")
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default="results/demo_run.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    questions = [args.question] if args.question else DEFAULT_QUESTIONS

    outputs = []
    for q in questions:
        print("\n" + "=" * 72)
        print("Q:", q)
        result = run_demo(args.pdf, q, cfg)
        outputs.append(result)
        if result.get("refused"):
            print("REFUSED:", result.get("refusal_reason"), "-", result.get("message"))
        else:
            print("A:", result.get("answer"))
        cites = (result.get("grounding") or {}).get("citations") or []
        if cites:
            print("Citations:", [c.get("page") for c in cites])
        pages = [p["page_number"] for p in result.get("retrieved_pages", [])]
        print("Retrieved pages:", pages)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"pdf": str(Path(args.pdf).resolve()), "results": outputs}, f, ensure_ascii=False, indent=2)
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
