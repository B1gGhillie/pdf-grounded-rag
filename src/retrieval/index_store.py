"""Persistent document index: page metadata, BM25 corpus, optional visual embeddings."""

import json
import pickle
import hashlib
from pathlib import Path


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
        from src.eval.provenance import atomic_json
        atomic_json(self.manifest_path, data)

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
        import torch

        torch.save(embeddings, self.embeddings_path)

    def load_visual_embeddings(self):
        if not self.embeddings_path.exists():
            return None
        import torch

        return torch.load(self.embeddings_path, weights_only=True, map_location="cpu")

    @staticmethod
    def fingerprint(pdf_path, cfg):
        from src.utils.text import TOKENIZER_VERSION
        from src.eval.provenance import file_sha256
        digest = file_sha256(pdf_path)
        settings = {"schema": 3, "pdf_sha256": digest, "pdf": cfg["pdf"],
                    "tokenizer": TOKENIZER_VERSION,
                    "models": {k: cfg["retrieval"].get(k) for k in
                               ("model", "clip_model", "model_revision", "clip_revision")}}
        tessdata = cfg["pdf"].get("tessdata")
        if tessdata:
            settings["ocr_data"] = {lang: file_sha256(Path(tessdata)/f"{lang}.traineddata")
                                    if (Path(tessdata)/f"{lang}.traineddata").exists() else None
                                    for lang in cfg["pdf"].get("ocr_language", "chi_sim+eng").split("+")}
        return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()

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
