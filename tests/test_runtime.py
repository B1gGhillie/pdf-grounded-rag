"""Tests for runtime backend selection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.runtime import choose_backends


def test_choose_backends_respects_explicit_config():
    cfg = {
        "runtime": {
            "visual_backend": "clip",
            "generation_backend": "extractive",
            "device": "cpu",
        }
    }
    backends = choose_backends(cfg)
    assert backends["visual_backend"] == "clip"
    assert backends["generation_backend"] == "extractive"
    assert backends["device"] == "cpu"


def test_demo_mode_forces_extractive():
    cfg = {
        "runtime": {
            "visual_backend": "colpali",
            "generation_backend": "vlm",
            "demo_mode": True,
        }
    }
    backends = choose_backends(cfg)
    assert backends["generation_backend"] == "extractive"
    assert backends["visual_backend"] in ("clip", "none")
