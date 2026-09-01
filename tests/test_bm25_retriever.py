"""Tests for BM25 retriever (no GPU required)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.bm25_retriever import BM25Retriever


def test_bm25_retrieve_relevant_page():
    page_paths = ["page_0.png", "page_1.png", "page_2.png"]
    text_chunks = [
        {"page": 1, "text": "Introduction to machine learning basics."},
        {"page": 2, "text": "Neural networks and deep learning architectures."},
        {"page": 3, "text": "Conclusion and future work."},
    ]
    retriever = BM25Retriever().build_index(page_paths, text_chunks)
    hits = retriever.retrieve("deep learning neural networks", top_k=2)

    assert len(hits) >= 1
    assert hits[0]["page_index"] == 1
    assert hits[0]["score"] > 0


def test_bm25_empty_page_handled():
    page_paths = ["p0.png", "p1.png"]
    text_chunks = [{"page": 1, "text": "Only page one has text."}]
    retriever = BM25Retriever().build_index(page_paths, text_chunks)
    densities = retriever.get_text_densities()

    assert densities[0] > 0
    assert densities[1] == 0
