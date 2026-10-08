"""Tests for runtime backend selection."""

import pytest

from src.utils import runtime


pytestmark = pytest.mark.unit


def _patch_runtime_probe(monkeypatch):
    monkeypatch.setattr(
        runtime,
        "probe_heavy_models",
        lambda: {
            "colpali": True,
            "vlm": True,
            "clip": True,
            "extractive": True,
            "free_ram_gb": 32.0,
            "free_vram_gb": 12.0,
            "reasons": [],
        },
    )
    monkeypatch.setattr(runtime, "resolve_device", lambda preferred: "cpu")


def test_choose_backends_respects_explicit_config(monkeypatch):
    _patch_runtime_probe(monkeypatch)
    cfg = {
        "runtime": {
            "visual_backend": "clip",
            "generation_backend": "extractive",
            "device": "cpu",
        }
    }
    backends = runtime.choose_backends(cfg)
    assert backends["visual_backend"] == "clip"
    assert backends["generation_backend"] == "extractive"
    assert backends["device"] == "cpu"


def test_demo_mode_forces_extractive_and_clip(monkeypatch):
    _patch_runtime_probe(monkeypatch)
    cfg = {
        "runtime": {
            "visual_backend": "colpali",
            "generation_backend": "vlm",
            "demo_mode": True,
        }
    }
    backends = runtime.choose_backends(cfg)
    assert backends["generation_backend"] == "extractive"
    assert backends["visual_backend"] == "clip"
