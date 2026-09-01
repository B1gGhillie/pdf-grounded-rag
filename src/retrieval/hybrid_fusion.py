"""Adaptive hybrid fusion of visual (ColPali) and text (BM25) retrieval scores.

Pages with high text density receive higher text-channel weight;
chart/scan-heavy pages rely more on the visual channel.
"""


def _normalize_scores(hits):
    if not hits:
        return {}
    scores = [h["score"] for h in hits]
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return {h["page_index"]: 1.0 for h in hits}
    return {
        h["page_index"]: (h["score"] - min_s) / (max_s - min_s) for h in hits
    }


def _page_channel_weights(text_density, max_density, min_text_w=0.25, max_text_w=0.75):
    """Map text density to per-page channel weights."""
    if max_density <= 0:
        return 0.5, 0.5
    ratio = text_density / max_density
    text_w = min_text_w + (max_text_w - min_text_w) * ratio
    return 1.0 - text_w, text_w


def _lookup_page_path(page_idx, *hit_lists):
    for hits in hit_lists:
        for h in hits:
            if h["page_index"] == page_idx:
                return h.get("page_path")
    return None


def fuse_retrieval(
    visual_hits,
    text_hits,
    page_text_densities=None,
    top_k=5,
    min_text_weight=0.25,
    max_text_weight=0.75,
):
    """Fuse visual and text retrieval with per-page adaptive weighting.

    Returns list of dicts sorted by fused_score descending:
        page_index, page_path, fused_score, visual_score, text_score,
        visual_weight, text_weight, text_snippet (if available)
    """
    visual_norm = _normalize_scores(visual_hits)
    text_norm = _normalize_scores(text_hits)
    all_pages = set(visual_norm) | set(text_norm)

    densities = page_text_densities or {}
    max_density = max(densities.values()) if densities else 0

    text_snippets = {h["page_index"]: h.get("text_snippet", "") for h in text_hits}

    fused = []
    for page_idx in all_pages:
        v_score = visual_norm.get(page_idx, 0.0)
        t_score = text_norm.get(page_idx, 0.0)
        density = densities.get(page_idx, 0)
        v_w, t_w = _page_channel_weights(
            density, max_density, min_text_weight, max_text_weight
        )
        combined = v_w * v_score + t_w * t_score

        fused.append(
            {
                "page_index": page_idx,
                "page_path": _lookup_page_path(page_idx, visual_hits, text_hits),
                "fused_score": combined,
                "visual_score": v_score,
                "text_score": t_score,
                "visual_weight": round(v_w, 3),
                "text_weight": round(t_w, 3),
                "text_snippet": text_snippets.get(page_idx, ""),
            }
        )

    fused.sort(key=lambda x: x["fused_score"], reverse=True)
    return fused[:top_k]
