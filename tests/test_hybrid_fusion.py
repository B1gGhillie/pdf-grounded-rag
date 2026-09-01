"""Tests for adaptive hybrid fusion (no GPU required)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.hybrid_fusion import fuse_retrieval


def test_fusion_prefers_text_on_dense_page():
    visual_hits = [
        {"page_index": 0, "page_path": "p0.png", "score": 0.9},
        {"page_index": 1, "page_path": "p1.png", "score": 0.5},
    ]
    text_hits = [
        {"page_index": 0, "page_path": "p0.png", "score": 0.2, "text_snippet": "sparse"},
        {"page_index": 1, "page_path": "p1.png", "score": 0.95, "text_snippet": "dense text"},
    ]
    densities = {0: 50, 1: 5000}

    fused = fuse_retrieval(visual_hits, text_hits, page_text_densities=densities, top_k=2)

    assert fused[0]["page_index"] == 1
    assert fused[0]["text_weight"] > fused[0]["visual_weight"]


def test_fusion_returns_both_channels_metadata():
    visual_hits = [{"page_index": 0, "page_path": "p0.png", "score": 0.8}]
    text_hits = [{"page_index": 0, "page_path": "p0.png", "score": 0.6, "text_snippet": "hello"}]

    fused = fuse_retrieval(visual_hits, text_hits, page_text_densities={0: 100}, top_k=1)

    assert "fused_score" in fused[0]
    assert "visual_score" in fused[0]
    assert "text_score" in fused[0]
    assert fused[0]["page_path"] == "p0.png"
