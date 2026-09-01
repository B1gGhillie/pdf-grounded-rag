"""Runtime backend selection and safe model probing."""

from __future__ import annotations

import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)


def resolve_device(preferred="auto"):
    import torch

    if preferred and preferred != "auto":
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"


def estimate_free_ram_gb():
    """Best-effort free RAM estimate (GB). Returns None if unknown."""
    try:
        import psutil

        return psutil.virtual_memory().available / (1024**3)
    except Exception:
        pass
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullAvailPhys / (1024**3)
    except Exception:
        return None


def estimate_free_vram_gb():
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        free, _total = torch.cuda.mem_get_info()
        return free / (1024**3)
    except Exception:
        return 0.0


@lru_cache(maxsize=1)
def probe_heavy_models():
    """Return which heavy backends are likely usable on this machine.

    ColPali (~3B) + Qwen2-VL-2B need substantial RAM/VRAM. On 6–8GB laptop
    GPUs with low system RAM, prefer CLIP + extractive fallbacks.
    """
    free_ram = estimate_free_ram_gb()
    free_vram = estimate_free_vram_gb()
    min_ram_for_colpali = float(os.environ.get("RAG_MIN_RAM_COLPALI", "12"))
    min_vram_for_vlm = float(os.environ.get("RAG_MIN_VRAM_VLM", "5.5"))

    colpali_ok = True
    vlm_ok = True
    reasons = []

    if free_ram is not None and free_ram < min_ram_for_colpali:
        colpali_ok = False
        reasons.append(
            "free_ram={0:.1f}GB < {1}GB (ColPali unsafe)".format(free_ram, min_ram_for_colpali)
        )
    if free_vram < min_vram_for_vlm:
        # Still allow VLM on CPU if RAM is large, but on this machine prefer extractive
        if free_ram is None or free_ram < 14:
            vlm_ok = False
            reasons.append(
                "free_vram={0:.1f}GB / free_ram insufficient for Qwen2-VL".format(free_vram)
            )

    return {
        "colpali": colpali_ok,
        "vlm": vlm_ok,
        "clip": True,
        "extractive": True,
        "free_ram_gb": free_ram,
        "free_vram_gb": free_vram,
        "reasons": reasons,
    }


def choose_backends(cfg):
    """Pick visual/generation backends from config + machine probe."""
    runtime = cfg.get("runtime", {})
    visual = (runtime.get("visual_backend") or "auto").lower()
    generation = (runtime.get("generation_backend") or "auto").lower()
    probe = probe_heavy_models()

    if visual == "auto":
        visual = "colpali" if probe["colpali"] else "clip"
    if generation == "auto":
        generation = "vlm" if probe["vlm"] else "extractive"

    # Forced demo mode
    if runtime.get("demo_mode"):
        visual = "none" if visual == "colpali" else visual
        if visual == "colpali":
            visual = "clip"
        generation = "extractive"

    return {
        "visual_backend": visual,
        "generation_backend": generation,
        "device": resolve_device(runtime.get("device", "auto")),
        "probe": probe,
    }
