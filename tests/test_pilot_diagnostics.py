import json

import pytest

from scripts.diagnose_benchmark import diagnose
from scripts.prepare_mmlongbench_pilot import prepare_pilot, select_pilot

pytestmark = pytest.mark.unit


def item(ident, *, answerable=True, pages=None, status="complete"):
    return {"id": ident, "doc_id": f"{ident}.pdf", "pdf_path": f"/{ident}.pdf",
            "document_sha256": ident, "question": "What is the value?",
            "answer": "5" if answerable else "", "answerable": answerable,
            "gold_pages": ([1] if answerable else []) if pages is None else pages,
            "evidence_annotation_status": status,
            "category": "text" if answerable else "unanswerable", "doc_type": "report"}


def test_pilot_balances_categories_and_keeps_complete_cross_page_questions():
    items = [item("single"), item("cross", pages=[1, 2]),
             item("negative", answerable=False, status="not_applicable"),
             item("long"), item("unlabeled", pages=[], status="missing")]
    docs = {i["pdf_path"]: {"pages": 120 if i["id"] == "long" else 20} for i in items}
    quotas = {"text": 2, "unanswerable": 1}
    chosen = select_pilot(items, docs, quotas=quotas)
    assert {i["id"] for i in chosen} == {"single", "cross", "negative"}
    assert chosen == select_pilot(list(reversed(items)), docs, quotas=quotas)
    assert next(i for i in chosen if i["id"] == "cross")["gold_pages"] == [1, 2]
    with pytest.raises(ValueError, match="Not enough"):
        select_pilot(items, docs, quotas={"text": 3})


@pytest.mark.parametrize("split", ["test", "full", "pilot_from_dev"])
def test_pilot_refuses_to_sample_non_dev_data(tmp_path, split):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps({"split": split, "items": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="requires a DEV"):
        prepare_pilot(path, tmp_path / "pilot.json")


def test_diagnosis_distinguishes_blocked_match_from_unanswerable_generation():
    items = [item("correct"), item("negative", answerable=False), item("verbose")]
    predictions = [
        {"id": "correct", "raw_answer": "5 [p.1]", "answer": None, "refused": True,
         "refusal_reason": "grounding_failed", "retrieved_pages": [{"page_index": 0}]},
        {"id": "negative", "raw_answer": "5", "answer": None, "refused": True},
        {"id": "verbose", "raw_answer": "The value is 5.", "answer": "The value is 5.",
         "retrieved_pages": [{"page_number": 1}]},
    ]
    report = diagnose({"status": "complete", "predictions": predictions}, items)
    assert report["summary"]["exact_matches_blocked"] == 1
    assert report["rows"][0]["triage"] == "exact_match_blocked"
    assert report["rows"][1]["triage"] == "unanswerable_candidate_generated"
    # Lexical mismatch is not sufficient to conclude semantic incorrectness.
    assert report["rows"][2]["triage"] == "inspect_generation_and_answer_format"
    assert report["rows"][2]["manual_review"]["raw_answer_correct"] is None


def test_incomplete_or_mismatched_results_are_not_presented_as_complete():
    items = [item("one"), item("two")]
    prediction = {"id": "one", "refused": True, "refusal_reason": "retrieval_miss"}
    report = diagnose({"status": "incomplete", "predictions": [prediction]}, items)
    assert report["status"] == "incomplete" and report["summary"]["expected_count"] == 2
    assert report["summary"]["count"] == 1
    assert report["rows"][0]["raw_lexical_em"] is None
    with pytest.raises(ValueError, match="every"):
        diagnose({"status": "complete", "predictions": [prediction]}, items)
    with pytest.raises(ValueError, match="Duplicate"):
        diagnose({"predictions": [prediction, prediction]}, items)
