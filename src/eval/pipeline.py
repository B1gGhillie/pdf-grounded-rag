"""Batch evaluation and ablation runner."""

import json
from pathlib import Path

from tqdm import tqdm

from src.eval.ablation import apply_ablation_config, list_ablation_configs
from src.eval.dataset_loader import load_eval_dataset
from src.eval.metrics import aggregate_metrics, score_prediction
from src.pipeline.rag_pipeline import run_rag
from src.pipeline.rag_retry import run_rag_with_retry


def _resolve_method(cfg, method_override=None):
    if method_override:
        return method_override
    return cfg.get("_method", "full")


def evaluate_predictions(predictions, gold_items):
    gold_by_id = {g["id"]: g for g in gold_items}
    metrics = []
    for pred in predictions:
        gold = gold_by_id.get(pred["id"])
        if gold is None:
            continue
        metrics.append(score_prediction(pred, gold))
    return metrics, aggregate_metrics(metrics)


def run_benchmark(
    dataset_path,
    cfg,
    method=None,
    output_path=None,
    force_rebuild=False,
    limit=None,
):
    items = load_eval_dataset(dataset_path)
    if limit is not None:
        items = items[:limit]

    predictions = []
    method = _resolve_method(cfg, method)
    
    # Use retry wrapper if retry is configured
    retry_enabled = cfg.get("retry", {}).get("max_retry", 0) > 0
    rag_fn = run_rag_with_retry if retry_enabled else run_rag

    for item in tqdm(items, desc="eval {0}".format(method)):
        pred = rag_fn(
            item["pdf_path"],
            item["question"],
            cfg,
            method=method,
            force_rebuild=force_rebuild,
        )
        pred["id"] = item["id"]
        predictions.append(pred)

    per_item, summary = evaluate_predictions(predictions, items)
    result = {
        "method": method,
        "dataset": str(dataset_path),
        "summary": summary,
        "predictions": predictions,
        "per_item_metrics": per_item,
    }

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def run_ablation_suite(dataset_path, base_cfg, output_dir, limit=None, force_rebuild=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suite_results = {}

    for name in list_ablation_configs():
        cfg = apply_ablation_config(base_cfg, name)
        method = cfg.get("_method", "full")
        out_path = output_dir / "{0}.json".format(name)
        suite_results[name] = run_benchmark(
            dataset_path,
            cfg,
            method=method,
            output_path=str(out_path),
            force_rebuild=force_rebuild,
            limit=limit,
        )

    summary_rows = {name: res["summary"] for name, res in suite_results.items()}
    summary_path = output_dir / "ablation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    return suite_results
