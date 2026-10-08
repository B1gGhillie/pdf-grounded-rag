"""Separate ranking from evidence confidence; query-aware reliability fusion."""
import math


def _normalize_scores(hits):
    """Legacy normalization used ONLY by the legacy_minmax ablation."""
    if not hits:
        return {}
    low, high = min(h["score"] for h in hits), max(h["score"] for h in hits)
    return {h["page_index"]: (h["score"]-low)/(high-low) if high > low else 1.0 for h in hits}


def _page_channel_weights(text_density, max_density, min_text_w=0.25, max_text_w=0.75):
    ratio = text_density / max_density if max_density > 0 else 0.5
    weight = min_text_w + (max_text_w-min_text_w) * ratio
    return 1-weight, weight


def absolute_score(raw, channel, visual_backend="clip", text_scale=8.0, visual_scale=20.0):
    raw = float(raw)
    if not math.isfinite(raw):
        raise ValueError("Non-finite retrieval score")
    raw = max(0.0, raw)
    if channel == "visual" and visual_backend == "clip":
        return min(1.0, raw)
    scale = text_scale if channel == "text" else visual_scale
    if scale <= 0:
        raise ValueError("Score scales must be positive")
    return raw / (raw + scale)


def fuse_retrieval(visual_hits, text_hits, page_text_densities=None, top_k=5,
                   min_text_weight=0.25, max_text_weight=0.75, mode="adaptive",
                   fixed_text_weight=0.5, visual_backend="clip", text_scale=8.0,
                   visual_scale=20.0, rrf_k=60, page_features=None,
                   use_query_coverage=True, use_text_quality=True):
    """reliability = density prior modulated by query coverage and text quality.

    Confidence is an uncalibrated monotone score, NOT an answer probability.
    fixed, adaptive (density only), RRF and legacy_minmax are control methods.
    """
    if mode not in {"reliability", "adaptive", "fixed", "rrf", "legacy_minmax"}:
        raise ValueError(f"Unknown fusion mode: {mode}")
    if not 0 <= min_text_weight <= max_text_weight <= 1 or not 0 <= fixed_text_weight <= 1:
        raise ValueError("Invalid fusion weights")
    if rrf_k <= 0 or top_k <= 0:
        raise ValueError("top_k and rrf_k must be positive")
    vs = {h["page_index"]: h for h in visual_hits}
    ts = {h["page_index"]: h for h in text_hits}
    vr = {h["page_index"]: i+1 for i, h in enumerate(visual_hits)}
    tr = {h["page_index"]: i+1 for i, h in enumerate(text_hits)}
    densities = page_text_densities or {}
    max_density = max(densities.values(), default=0)
    legacy_v, legacy_t = _normalize_scores(visual_hits), _normalize_scores(text_hits)
    fused = []
    for idx in sorted(set(vs) | set(ts)):
        v, t = vs.get(idx, {}), ts.get(idx, {})
        features = (page_features or {}).get(idx, t)
        v_score = absolute_score(v.get("score", 0), "visual", visual_backend, text_scale, visual_scale)
        t_score = absolute_score(t.get("score", 0), "text", visual_backend, text_scale, visual_scale)
        _, prior = _page_channel_weights(densities.get(idx, 0), max_density,
                                         min_text_weight, max_text_weight)
        quality = max(0.0, min(1.0, float(features.get("text_quality", 1.0))))
        coverage = max(0.0, min(1.0, float(features.get("query_coverage", 0.0))))
        if mode == "reliability":
            tw = min_text_weight + (prior-min_text_weight) * (quality if use_text_quality else 1) * (coverage if use_query_coverage else 1)
        elif mode in {"fixed", "rrf"}:
            tw = fixed_text_weight
        else:
            tw = prior
        if not visual_hits:
            tw = 1.0
        elif not text_hits:
            tw = 0.0
        confidence = (1-tw)*v_score + tw*t_score
        if mode == "rrf":
            score = (1/(rrf_k+vr[idx]) if idx in vr else 0) + (1/(rrf_k+tr[idx]) if idx in tr else 0)
        elif mode == "legacy_minmax":
            score = (1-tw)*legacy_v.get(idx, 0) + tw*legacy_t.get(idx, 0)
        else:
            score = confidence
        fused.append({"page_index": idx, "page_path": v.get("page_path") or t.get("page_path"),
                      "fused_score": score, "evidence_confidence": confidence,
                      "visual_score": v_score, "text_score": t_score,
                      "raw_visual_score": v.get("score", 0), "raw_text_score": t.get("score", 0),
                      "visual_weight": 1-tw, "text_weight": tw, "query_coverage": coverage,
                      "text_quality": quality, "fusion_mode": mode,
                      "text_snippet": t.get("text_snippet", ""), "text": features.get("text", ""),
                      "text_source": features.get("text_source", "unknown")})
    return sorted(fused, key=lambda h: (-h["fused_score"], h["page_index"]))[:top_k]
