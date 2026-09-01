"""Run benchmark or ablation suite on a JSON dataset."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.ablation import apply_ablation_config, list_ablation_configs
from src.eval.pipeline import run_ablation_suite, run_benchmark
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Benchmark / ablation runner")
    parser.add_argument("--dataset", type=str, default="data/eval/sample_dataset.json")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--method",
        type=str,
        default="full",
        choices=["full", "hybrid", "visual", "ocr", "demo"],
    )
    parser.add_argument("--ablation", type=str, default=None, help="Run named ablation config")
    parser.add_argument("--ablation-all", action="store_true", help="Run all ablation configs")
    parser.add_argument(
        "--ablation-demo",
        action="store_true",
        help="Run laptop-safe demo_* ablations only",
    )
    parser.add_argument("--output", type=str, default="results/benchmark.json")
    parser.add_argument("--output-dir", type=str, default="results/ablation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--laptop", action="store_true", help="Use configs/laptop.yaml")
    args = parser.parse_args()

    cfg = load_config("configs/laptop.yaml" if args.laptop else args.config)
    if args.laptop or args.method == "demo":
        cfg.setdefault("runtime", {})["demo_mode"] = True
        cfg["runtime"]["visual_backend"] = cfg["runtime"].get("visual_backend") or "clip"
        cfg["runtime"]["generation_backend"] = "extractive"

    if args.ablation_demo:
        from src.eval.ablation import ABLATION_CONFIGS, apply_ablation_config as _apply
        from src.eval.pipeline import run_benchmark as _run

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        suite = {}
        for name in [k for k in ABLATION_CONFIGS if k.startswith("demo_")]:
            acfg = _apply(cfg, name)
            method = acfg.get("_method", "full")
            suite[name] = _run(
                args.dataset,
                acfg,
                method=method,
                output_path=str(output_dir / "{0}.json".format(name)),
                force_rebuild=args.force_rebuild,
                limit=args.limit,
            )
        summary_rows = {name: res["summary"] for name, res in suite.items()}
        with open(output_dir / "ablation_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_rows, f, ensure_ascii=False, indent=2)
        print(json.dumps(summary_rows, ensure_ascii=False, indent=2))
        return

    if args.ablation_all:
        suite = run_ablation_suite(
            args.dataset,
            cfg,
            args.output_dir,
            limit=args.limit,
            force_rebuild=args.force_rebuild,
        )
        print(json.dumps({k: v["summary"] for k, v in suite.items()}, ensure_ascii=False, indent=2))
        return

    if args.ablation:
        cfg = apply_ablation_config(cfg, args.ablation)
        method = cfg.get("_method", args.method)
    else:
        method = args.method

    result = run_benchmark(
        args.dataset,
        cfg,
        method=method,
        output_path=args.output,
        force_rebuild=args.force_rebuild,
        limit=args.limit,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
