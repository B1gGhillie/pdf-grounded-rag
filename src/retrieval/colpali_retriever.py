"""ColPali-based visual page retrieval with optional embedding cache."""

from __future__ import annotations

import logging

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class ColPaliRetriever:
    def __init__(self, model_name="vidore/colpali-v1.3-hf", device=None, torch_dtype=None):
        from transformers import ColPaliForRetrieval, ColPaliProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        self.processor = ColPaliProcessor.from_pretrained(model_name)
        # Load weights straight onto target device when possible to cut CPU RAM spike.
        try:
            self.model = ColPaliForRetrieval.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                device_map=self.device if self.device != "cpu" else None,
            )
            if self.device == "cpu":
                self.model = self.model.to(self.device)
        except Exception as exc:
            logger.warning("ColPali device_map load failed (%s); retrying plain .to()", exc)
            self.model = ColPaliForRetrieval.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed_pages(self, page_image_paths):
        """Pre-compute visual embeddings for all pages (cacheable)."""
        images = [Image.open(p).convert("RGB") for p in page_image_paths]
        inputs = self.processor(images=images).to(self.device)
        return self.model(**inputs).embeddings

    @torch.no_grad()
    def _score_pages(self, query, page_image_paths, cached_embeddings=None):
        inputs_text = self.processor(text=[query]).to(self.device)
        query_embeddings = self.model(**inputs_text).embeddings

        if cached_embeddings is not None:
            image_embeddings = cached_embeddings.to(self.device)
        else:
            images = [Image.open(p).convert("RGB") for p in page_image_paths]
            inputs_images = self.processor(images=images).to(self.device)
            image_embeddings = self.model(**inputs_images).embeddings

        return self.processor.score_retrieval(query_embeddings, image_embeddings)[0]

    @torch.no_grad()
    def retrieve(self, query, page_image_paths, top_k=5, cached_embeddings=None):
        scores = self._score_pages(query, page_image_paths, cached_embeddings)
        ranked = sorted(enumerate(scores.tolist()), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            {"page_index": idx, "page_path": page_image_paths[idx], "score": float(score)}
            for idx, score in ranked
        ]
