"""Full RAG: Hybrid Fusion + Grounding + Multi-signal Refusal."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.rag_retry import run_rag_with_retry
from src.eval.provenance import atomic_json
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Full RAG pipeline")
    parser.add_argument("--pdf", type=str, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--method",
        type=str,
        default="full",
        choices=["full", "hybrid", "visual", "ocr", "text"],
        help="Pipeline variant",
    )
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--output", help="Save complete evidence and verification trace as JSON")
    parser.add_argument("--max-retry", type=int, help="Override bounded retry count")
    parser.add_argument("--calibration", help="Frozen dev threshold artifact (single-pass)")
    parser.add_argument("--no-grounding", action="store_true")
    parser.add_argument("--no-refusal", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Laptop-safe path: CLIP/BM25 + extractive answers (no ColPali/VLM)",
    )
    parser.add_argument(
        "--config-laptop",
        action="store_true",
        help="Use configs/laptop.yaml",
    )
    args = parser.parse_args()

    config_path = "configs/laptop.yaml" if args.config_laptop else args.config
    cfg = load_config(config_path)
    if args.max_retry is not None:
        cfg.setdefault("retry", {})["max_retry"] = args.max_retry
    if args.calibration:
        cfg.setdefault("refusal", {})["calibration_path"] = args.calibration
    if args.demo or args.config_laptop:
        cfg.setdefault("runtime", {})["demo_mode"] = True
        cfg["runtime"]["visual_backend"] = cfg["runtime"].get("visual_backend") or "clip"
        cfg["runtime"]["generation_backend"] = "extractive"
        cfg.setdefault("grounding", {})["verify_mode"] = "lexical"
    if args.no_grounding:
        cfg.setdefault("grounding", {})["enabled"] = False
    if args.no_refusal:
        cfg.setdefault("refusal", {})["enabled"] = False
    if args.no_verify:
        cfg.setdefault("grounding", {})["verify"] = False
        cfg.setdefault("refusal", {})["use_grounding_gate"] = False

    method = "demo" if args.demo else args.method
    result = run_rag_with_retry(
        args.pdf,
        args.question,
        cfg,
        method=method,
        force_rebuild=args.force_rebuild,
    )
    if args.output:
        atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
