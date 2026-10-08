"""Fit a retrieval gate threshold on dev documents, never on test labels.

This calibrates a decision threshold, NOT a probability or conformal guarantee.
"""
import math
from pathlib import Path

from src.eval.provenance import json_hash, file_sha256
from src.utils.text import TOKENIZER_VERSION


def score_signature(cfg, method, visual_backend):
    pdf = cfg.get("pdf", {})
    ocr_data = {}
    if pdf.get("tessdata"):
        for language in pdf.get("ocr_language", "chi_sim+eng").split("+"):
            path = Path(pdf["tessdata"]) / f"{language}.traineddata"
            ocr_data[language] = file_sha256(path) if path.exists() else None
    return json_hash({"schema": 1, "tokenizer": TOKENIZER_VERSION,
                      "retrieval": cfg.get("retrieval", {}), "pdf": cfg.get("pdf", {}),
                      "method": "full" if method == "demo" else method,
                      "visual_backend": visual_backend, "ocr_data": ocr_data})


def fit_threshold(scores, answerable, max_false_refusal_rate=0.1, has_evidence=None):
    if len(scores) != len(answerable) or not scores:
        raise ValueError("Nonempty aligned scores and labels are required")
    if any(type(y) is not bool for y in answerable) or not any(answerable) or all(answerable):
        raise ValueError("Calibration requires both answerable and unanswerable dev examples")
    if not 0 <= max_false_refusal_rate <= 1:
        raise ValueError("max_false_refusal_rate must be in [0,1]")
    if any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
        raise ValueError("Evidence confidence must be finite and in [0,1]")
    has_evidence = [True] * len(scores) if has_evidence is None else has_evidence
    if len(has_evidence) != len(scores) or any(type(v) is not bool for v in has_evidence):
        raise ValueError("has_evidence must contain one boolean per score")
    # Match the runtime decision s < threshold, including tied score groups.
    candidates = sorted({0.0, *(math.nextafter(s, math.inf) for s in scores)})
    rows = []
    for threshold in candidates:
        refused = [not found or s < threshold for s, found in zip(scores, has_evidence)]
        tp = sum(r and not y for r, y in zip(refused, answerable))
        fp = sum(r and y for r, y in zip(refused, answerable))
        fn = sum(not r and not y for r, y in zip(refused, answerable))
        rows.append({"threshold": threshold, "refusal_f1": 2*tp/(2*tp+fp+fn),
                     "false_refusal_rate": fp/sum(answerable), "tp": tp, "fp": fp, "fn": fn})
    feasible = [r for r in rows if r["false_refusal_rate"] <= max_false_refusal_rate]
    if not feasible:
        raise ValueError("No feasible threshold: missing retrieval evidence already exceeds the dev false-refusal constraint")
    best = max(feasible, key=lambda r: (r["refusal_f1"], -r["false_refusal_rate"], -r["threshold"]))
    return {"threshold": best["threshold"], "dev_metrics": best, "sweep": rows,
            "max_false_refusal_rate": max_false_refusal_rate,
            "note": "Dev-set constraint only; no test error guarantee or probability calibration."}


def validate_artifact(artifact, signature, test_document_hashes=None):
    if artifact.get("schema") != 1 or artifact.get("score_signature") != signature:
        raise ValueError("Calibration signature mismatch: refit on dev with this retrieval configuration")
    if not artifact.get("dev_document_sha256"):
        raise ValueError("Calibration artifact must record dev document hashes")
    threshold = artifact.get("threshold")
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Invalid calibration threshold")
    overlap = set(artifact.get("dev_document_sha256", [])) & set(test_document_hashes or [])
    if overlap:
        raise ValueError("Calibration/test PDF leakage detected (identical document content)")
    return threshold
