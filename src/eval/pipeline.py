"""Batch evaluation and ablation runner."""

import json
import copy
from pathlib import Path

from tqdm import tqdm

from src.eval.ablation import apply_ablation_config, list_ablation_configs
from src.eval.dataset_loader import load_eval_dataset, audit_documents
from src.eval.metrics import aggregate_metrics, score_prediction, risk_coverage_curve, document_bootstrap_ci
from src.eval.provenance import collect_provenance, atomic_json
from src.pipeline.rag_pipeline import run_rag
from src.pipeline.rag_retry import run_rag_with_retry


def _resolve_method(cfg, method_override=None):
    if method_override:
        return method_override
    return cfg.get("_method", "full")


def evaluate_predictions(predictions, gold_items):
    gold_by_id = {g["id"]: g for g in gold_items}
    if len(gold_by_id) != len(gold_items) or len({p["id"] for p in predictions}) != len(predictions):
        raise ValueError("Duplicate prediction or gold IDs")
    if {p["id"] for p in predictions} != set(gold_by_id):
        raise ValueError("Predictions and gold IDs must match exactly; no silently dropped examples")
    metrics = []
    for pred in predictions:
        gold = gold_by_id.get(pred["id"])
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
    cfg = copy.deepcopy(cfg)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        items = items[:limit]
    audit = audit_documents(items, cfg.get("pdf", {}).get("max_pages_per_doc"))
    calibration = cfg.get("refusal", {}).get("calibration_path")
    if calibration:
        artifact = json.loads(Path(calibration).read_text(encoding="utf-8"))
        if set(artifact.get("dev_document_sha256", [])) & {i["document_sha256"] for i in items}:
            raise ValueError("Calibration/test PDF leakage detected")
    seed = cfg.get("evaluation", {}).get("seed", 42)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    provenance = collect_provenance(cfg, dataset_path, items)

    predictions = []
    method = _resolve_method(cfg, method)
    
    # Use retry wrapper if retry is configured
    retry_enabled = cfg.get("retry", {}).get("max_retry", 0) > 0
    rag_fn = run_rag_with_retry if retry_enabled else run_rag

    rebuilt = set()
    for item in tqdm(items, desc="eval {0}".format(method)):
        pred = rag_fn(
            item["pdf_path"],
            item["question"],
            cfg,
            method=method,
            force_rebuild=force_rebuild and item["pdf_path"] not in rebuilt,
        )
        rebuilt.add(item["pdf_path"])
        pred["id"] = item["id"]
        predictions.append(pred)
        if output_path:
            atomic_json(str(output_path) + ".partial.json", {"status": "incomplete", "provenance": provenance, "predictions": predictions})

    per_item, summary = evaluate_predictions(predictions, items)
    result = {
        "method": method,
        "dataset": str(dataset_path),
        "summary": summary,
        "predictions": predictions,
        "per_item_metrics": per_item,
        "status": "complete",
        "provenance": provenance,
        "document_audit": audit,
        "risk_coverage": risk_coverage_curve(per_item),
        "confidence_intervals": document_bootstrap_ci(per_item, seed=seed),
        "by_category": {category: aggregate_metrics([m for m, item in zip(per_item, items) if item.get("category", "unspecified") == category])
                        for category in sorted({i.get("category", "unspecified") for i in items})},
    }

    if output_path:
        atomic_json(output_path, result)
        Path(str(output_path) + ".partial.json").unlink(missing_ok=True)

    return result


def run_ablation_suite(dataset_path, base_cfg, output_dir, limit=None, force_rebuild=False, suite="research"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suite_results = {}

    for name in list_ablation_configs(suite):
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
