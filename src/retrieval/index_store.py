"""Persistent document index: page metadata, BM25 corpus, optional visual embeddings."""

import json
import pickle
from pathlib import Path

import torch


class DocumentIndex:
    """Per-document cache under data/cache/{doc_id}/."""

    def __init__(self, doc_id, cache_dir="data/cache"):
        self.doc_id = doc_id
        self.root = Path(cache_dir) / doc_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.bm25_path = self.root / "bm25_index.pkl"
        self.embeddings_path = self.root / "visual_embeddings.pt"

    def save_manifest(self, data):
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_manifest(self):
        if not self.manifest_path.exists():
            return None
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_bm25(self, bm25_data):
        with open(self.bm25_path, "wb") as f:
            pickle.dump(bm25_data, f)

    def load_bm25(self):
        if not self.bm25_path.exists():
            return None
        with open(self.bm25_path, "rb") as f:
            return pickle.load(f)

    def save_visual_embeddings(self, embeddings):
        torch.save(embeddings, self.embeddings_path)

    def load_visual_embeddings(self):
        if not self.embeddings_path.exists():
            return None
        return torch.load(self.embeddings_path, weights_only=True)

    def is_fresh(self, pdf_path, dpi, max_pages):
        """Return True if cached index matches current PDF settings."""
        manifest = self.load_manifest()
        if manifest is None:
            return False
        pdf_stat = Path(pdf_path).stat()
        return (
            manifest.get("pdf_path") == str(Path(pdf_path).resolve())
            and manifest.get("pdf_mtime") == pdf_stat.st_mtime
            and manifest.get("dpi") == dpi
            and manifest.get("max_pages") == max_pages
        )
