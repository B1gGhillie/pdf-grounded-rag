import copy
import pytest
from src.eval.metrics import exact_match, token_f1, precision_at_k, score_prediction, aggregate_metrics, risk_coverage_curve, document_bootstrap_ci
from src.eval.dataset_loader import validate_items, audit_documents, split_by_document
from src.eval.pipeline import evaluate_predictions
from src.eval.ablation import apply_ablation_config, list_ablation_configs
from src.refusal.calibration import fit_threshold, validate_artifact

pytestmark = pytest.mark.unit


def test_metrics_preserve_decimal_units_chinese_and_token_multiplicity():
    assert exact_match("0.84", "84") == 0
    assert exact_match("350万元", "350亿元") == 0
    assert token_f1("a a b", "a b b") == pytest.approx(2/3)
    assert token_f1("节水工程", "节水项目") == .5
    assert precision_at_k([1], [1], 5) == .2


def test_false_refusal_is_zero_and_rejected_citations_are_not_delivered():
    pred = {"refused": True, "answer": None, "raw_answer": "350万元[p.1]", "grounding": {"citations": [{"page": 1}]}}
    metric = score_prediction(pred, {"id": "q", "answerable": True, "answer": "350万元", "gold_pages": [1]})
    assert metric["f1"] == 0 and metric["citation_f1"] == 0 and metric["conditional_f1"] is None
    assert aggregate_metrics([metric])["f1"] == 0


def test_model_refusal_counted_even_when_policy_disabled():
    row = score_prediction({"answer": "UNANSWERABLE", "refused": False}, {"id": "q", "answerable": False})
    assert row["true_positive_refuse"] == 1
    bad = score_prediction({"answer": "猜测", "refused": False}, {"id": "q", "answerable": False})
    assert aggregate_metrics([bad])["refusal_f1"] == 0


def test_curve_ties_and_single_document_uncertainty():
    rows = [{"candidate_correct": 1, "confidence": .8, "doc_id": "a"},
            {"candidate_correct": 0, "confidence": .8, "doc_id": "a"},
            {"candidate_correct": None, "confidence": .1, "doc_id": "a"}]
    curve = risk_coverage_curve(rows)
    assert curve["count"] == 2 and len(curve["points"]) == 1
    assert curve["points"][0]["risk"] == .5
    assert document_bootstrap_ci(rows)["intervals"] is None


def test_calibration_ties_constraints_and_leakage():
    result = fit_threshold([.1, .1, .8, .9], [False, False, True, True], 0)
    assert .1 < result["threshold"] < .8
    assert result["dev_metrics"]["refusal_f1"] == 1
    artifact = {"schema": 1, "score_signature": "sig", "dev_document_sha256": ["abc"], **result}
    assert validate_artifact(artifact, "sig", ["def"]) == result["threshold"]
    with pytest.raises(ValueError, match="leakage"):
        validate_artifact(artifact, "sig", ["abc"])
    with pytest.raises(ValueError, match="signature"):
        validate_artifact(artifact, "other")
    with pytest.raises(ValueError):
        fit_threshold([.5], [True])
    missing = fit_threshold([0, .8], [False, True], has_evidence=[False, True])
    assert missing["threshold"] == 0 and missing["dev_metrics"]["refusal_f1"] == 1
    with pytest.raises(ValueError, match="No feasible threshold"):
        fit_threshold([0, .8], [True, False], 0, [False, True])


def test_ablation_preserves_nested_parameters_and_isolates_retry():
    cfg = {"retrieval": {"top_k": 3, "fusion": {"mode": "reliability", "text_scale": 11}}, "retry": {"max_retry": 2}}
    before = copy.deepcopy(cfg)
    result = apply_ablation_config(cfg, "rrf_fusion")
    assert result["retrieval"]["fusion"] == {"mode": "rrf", "text_scale": 11}
    assert result["retry"]["max_retry"] == 0 and cfg == before
    assert not any(n.startswith("demo_") for n in list_ablation_configs("research"))
    cfg["evaluation"] = {"disable_retrieval_gate": True}
    assert apply_ablation_config(cfg, "full")["refusal"]["use_retrieval_gate"] is False


def test_no_silent_dropped_predictions_or_invalid_labels():
    with pytest.raises(ValueError, match="match exactly"):
        evaluate_predictions([], [{"id": "q"}])
    with pytest.raises(ValueError, match="boolean"):
        validate_items([{"id": "q", "question": "?", "answerable": "false", "answer": "", "gold_pages": []}])


def test_document_split_uses_content_and_rejects_truncation(tmp_path):
    import pymupdf
    items = []
    for name in ("one", "two"):
        path = tmp_path / f"{name}.pdf"
        with pymupdf.open() as doc:
            doc.new_page().insert_text((30, 30), name)
            doc.new_page()
            doc.save(path)
        items.append({"id": name, "pdf_path": str(path), "question": "?", "answerable": True, "answer": name, "gold_pages": [2]})
    copy_path = tmp_path / "renamed.pdf"
    copy_path.write_bytes((tmp_path / "one.pdf").read_bytes())
    items.append({**items[0], "id": "renamed", "pdf_path": str(copy_path)})
    splits = split_by_document(items)
    assert not ({i["document_sha256"] for i in splits["dev"]} & {i["document_sha256"] for i in splits["test"]})
    with pytest.raises(ValueError, match="max_pages_per_doc"):
        audit_documents(items, max_pages=1)
