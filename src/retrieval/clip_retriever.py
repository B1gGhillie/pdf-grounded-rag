"""Lightweight CLIP visual page retriever (laptop-friendly ColPali alternative)."""

from __future__ import annotations

import torch
from PIL import Image

_CLIP_CACHE = {}


def _as_embed_tensor(feats):
    """Normalize transformers version differences for CLIP feature outputs."""
    if torch.is_tensor(feats):
        return feats
    if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
        return feats.pooler_output
    if hasattr(feats, "last_hidden_state"):
        return feats.last_hidden_state[:, 0]
    raise TypeError("Unsupported CLIP feature type: {0}".format(type(feats)))


def _load_clip(model_name, device):
    key = (model_name, device)
    if key in _CLIP_CACHE:
        return _CLIP_CACHE[key]
    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(model_name)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = CLIPModel.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device).eval()
    _CLIP_CACHE[key] = (model, processor, dtype)
    return _CLIP_CACHE[key]


class CLIPRetriever:
    """Encode page images + queries with CLIP and rank by cosine similarity."""

    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.processor, self.dtype = _load_clip(model_name, self.device)

    @torch.no_grad()
    def embed_pages(self, page_image_paths, batch_size=8):
        vectors = []
        for i in range(0, len(page_image_paths), batch_size):
            batch_paths = page_image_paths[i : i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            pixel_values = inputs["pixel_values"].to(self.device)
            feats = _as_embed_tensor(self.model.get_image_features(pixel_values=pixel_values))
            feats = torch.nn.functional.normalize(feats.float(), dim=-1)
            vectors.append(feats.cpu())
        return torch.cat(vectors, dim=0)

    @torch.no_grad()
    def _embed_query(self, query):
        inputs = self.processor(
            text=[query], return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = _as_embed_tensor(self.model.get_text_features(**inputs))
        return torch.nn.functional.normalize(feats.float(), dim=-1).cpu()

    @torch.no_grad()
    def retrieve(self, query, page_image_paths, top_k=5, cached_embeddings=None):
        if cached_embeddings is None:
            image_emb = self.embed_pages(page_image_paths)
        else:
            image_emb = cached_embeddings.float()
            if image_emb.dim() > 2:
                # Defensive: ColPali multi-vector cache is incompatible; re-embed
                image_emb = self.embed_pages(page_image_paths)

        query_emb = self._embed_query(query)  # [1, D]
        scores = (image_emb @ query_emb.T).squeeze(-1)  # [N]
        ranked = sorted(enumerate(scores.tolist()), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {
                "page_index": idx,
                "page_path": page_image_paths[idx],
                "score": float(score),
            }
            for idx, score in ranked
        ]
