"""Fit a frozen retrieval threshold from an ungated DEV benchmark result."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.refusal.calibration import fit_threshold
from src.eval.provenance import atomic_json, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-false-refusal-rate", type=float, default=0.1)
    args = parser.parse_args()
    run = json.loads(Path(args.dev_results).read_text(encoding="utf-8"))
    if run.get("status") != "complete":
        raise ValueError("Calibration requires a completed benchmark")
    provenance = run["provenance"]
    cfg = provenance["config"]
    if cfg.get("retry", {}).get("max_retry", 0):
        raise ValueError("Use a single-pass dev run for threshold fitting")
    signatures = {p["score_signature"] for p in run["predictions"]}
    if len(signatures) != 1:
        raise ValueError("Dev examples used different retrieval configurations")
    rows = run["per_item_metrics"]
    found = {p["id"]: bool(p.get("retrieved_pages")) for p in run["predictions"]}
    fit = fit_threshold([r["confidence"] for r in rows], [not r["gold_refused"] for r in rows],
                        args.max_false_refusal_rate, [found[r["id"]] for r in rows])
    artifact = {"schema": 1, "kind": "retrieval_threshold_only", "score_signature": signatures.pop(),
                "dev_document_sha256": provenance["document_sha256"],
                "dev_results_sha256": file_sha256(args.dev_results), "dev_count": len(rows), **fit}
    atomic_json(args.output, artifact)
    print(json.dumps({k: v for k, v in artifact.items() if k != "sweep"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
