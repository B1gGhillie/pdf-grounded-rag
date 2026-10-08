"""Metadata cache tests that deliberately do not require PyTorch."""

import pytest

from src.retrieval.index_store import DocumentIndex


pytestmark = pytest.mark.unit


def test_manifest_and_bm25_cache_work_without_visual_embeddings(tmp_path):
    index = DocumentIndex("sample", cache_dir=tmp_path)
    manifest = {"pdf_path": "sample.pdf", "dpi": 120, "max_pages": 5}
    bm25 = {"documents": ["grounded retrieval"]}

    index.save_manifest(manifest)
    index.save_bm25(bm25)

    assert index.load_manifest() == manifest
    assert index.load_bm25() == bm25
