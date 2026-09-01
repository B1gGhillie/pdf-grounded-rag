"""Hybrid RAG pipeline (wrapper around unified run_rag)."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.rag_pipeline import run_rag
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Hybrid RAG: ColPali + BM25 fusion")
    parser.add_argument("--pdf", type=str, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--no-grounding", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.no_grounding:
        cfg.setdefault("grounding", {})["enabled"] = False
    cfg.setdefault("refusal", {})["enabled"] = False

    result = run_rag(
        args.pdf,
        args.question,
        cfg,
        method="hybrid",
        force_rebuild=args.force_rebuild,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
