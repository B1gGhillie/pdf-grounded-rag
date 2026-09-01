"""Create a richer sample PDF and self-built evaluation dataset for local debugging.

Covers: factual QA, numeric lookup, cross-page reasoning, table-like content,
and multiple unanswerable / out-of-scope questions for refusal evaluation.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf


def create_sample_pdf(pdf_path):
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open()
    pages = [
        (
            "Visual RAG for Complex PDFs\n\n"
            "This document describes a multimodal retrieval-augmented generation system "
            "for long PDF documents. The main contribution is page-level grounding.\n\n"
            "Motivation: complex PDFs mix text, tables, and figures. Pure OCR pipelines "
            "lose layout, while vision-only pipelines may ignore dense text."
        ),
        (
            "Adaptive Hybrid Fusion\n\n"
            "The system combines ColPali visual retrieval with BM25 text retrieval. "
            "Pages with high text density receive higher text-channel weight.\n\n"
            "Fusion formula: fused = w_v * s_v + w_t * s_t, where w_t grows with "
            "normalized text density in [0.25, 0.75]."
        ),
        (
            "Evaluation Results\n\n"
            "On the sample benchmark, hybrid RAG achieved 87.3% answer accuracy. "
            "Refusal precision reached 90.1% on unanswerable questions.\n\n"
            "Page recall@5 for hybrid retrieval was 0.84."
        ),
        (
            "System Components\n\n"
            "Component | Role\n"
            "---------------------------\n"
            "ColPali   | Visual page retrieval\n"
            "BM25      | Text page retrieval\n"
            "Qwen2-VL  | Multimodal answer generation\n"
            "Verifier  | Cite-then-Verify page check\n\n"
            "The verifier asks YES/NO whether each cited page supports the claim."
        ),
        (
            "Limitations and Scope\n\n"
            "This report only evaluates English and Chinese technical PDFs under "
            "50 pages. It does not discuss stock prices, celebrity biographies, "
            "or real-time news events. Training details for ColPali are omitted."
        ),
    ]

    for text in pages:
        page = doc.new_page()
        rect = pymupdf.Rect(54, 54, page.rect.width - 54, page.rect.height - 54)
        # insert_textbox wraps lines; insert_text silently clips overflow.
        font_size = 11
        while font_size >= 9:
            overflow = page.insert_textbox(
                rect, text, fontsize=font_size, align=pymupdf.TEXT_ALIGN_LEFT
            )
            if overflow >= 0:
                break
            page.add_redact_annot(page.rect)
            page.apply_redactions()
            font_size -= 1

    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path.resolve())


def create_sample_dataset(pdf_path, dataset_path):
    """Write dataset with relative pdf_path for portability."""
    dataset_path = Path(dataset_path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    # Relative to data/eval/ → ../pdfs/sample_doc.pdf
    pdf_rel = "../pdfs/{0}".format(Path(pdf_path).name)

    items = [
        {
            "id": "q1",
            "pdf_path": pdf_rel,
            "question": "What is the main contribution of this document?",
            "answer": "page-level grounding",
            "gold_pages": [1],
            "answerable": True,
            "category": "factual",
        },
        {
            "id": "q2",
            "pdf_path": pdf_rel,
            "question": "What accuracy did hybrid RAG achieve?",
            "answer": "87.3%",
            "gold_pages": [3],
            "answerable": True,
            "category": "numeric",
        },
        {
            "id": "q3",
            "pdf_path": pdf_rel,
            "question": "Who is the CEO of OpenAI?",
            "answer": "",
            "gold_pages": [],
            "answerable": False,
            "category": "out_of_scope",
        },
        {
            "id": "q4",
            "pdf_path": pdf_rel,
            "question": "Which model is used for multimodal answer generation?",
            "answer": "Qwen2-VL",
            "gold_pages": [4],
            "answerable": True,
            "category": "table_lookup",
        },
        {
            "id": "q5",
            "pdf_path": pdf_rel,
            "question": "What is the text-channel weight range used in fusion?",
            "answer": "[0.25, 0.75]",
            "gold_pages": [2],
            "answerable": True,
            "category": "numeric",
        },
        {
            "id": "q6",
            "pdf_path": pdf_rel,
            "question": "What page recall@5 did hybrid retrieval achieve?",
            "answer": "0.84",
            "gold_pages": [3],
            "answerable": True,
            "category": "numeric",
        },
        {
            "id": "q7",
            "pdf_path": pdf_rel,
            "question": "What was Apple's closing stock price yesterday?",
            "answer": "",
            "gold_pages": [],
            "answerable": False,
            "category": "out_of_scope",
        },
        {
            "id": "q8",
            "pdf_path": pdf_rel,
            "question": "How does the verifier check cited pages?",
            "answer": "YES/NO whether each cited page supports the claim",
            "gold_pages": [4],
            "answerable": True,
            "category": "factual",
        },
        {
            "id": "q9",
            "pdf_path": pdf_rel,
            "question": "Why do pure OCR pipelines struggle on complex PDFs?",
            "answer": "lose layout",
            "gold_pages": [1],
            "answerable": True,
            "category": "factual",
        },
        {
            "id": "q10",
            "pdf_path": pdf_rel,
            "question": "What training hyperparameters were used for ColPali in this report?",
            "answer": "",
            "gold_pages": [],
            "answerable": False,
            "category": "insufficient_evidence",
        },
    ]

    payload = {
        "name": "self_built_sample",
        "description": "Self-built mini benchmark for page grounding and refusal",
        "items": items,
    }
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(dataset_path.resolve())


def main():
    root = Path(__file__).resolve().parent.parent
    pdf_path = create_sample_pdf(root / "data" / "pdfs" / "sample_doc.pdf")
    dataset_path = create_sample_dataset(
        pdf_path, root / "data" / "eval" / "sample_dataset.json"
    )
    # Also write a named self-built copy for papers / experiment scripts
    self_built = create_sample_dataset(
        pdf_path, root / "data" / "eval" / "self_built_dataset.json"
    )
    print("Created sample PDF:", pdf_path)
    print("Created sample dataset:", dataset_path)
    print("Created self-built dataset:", self_built)


if __name__ == "__main__":
    main()
