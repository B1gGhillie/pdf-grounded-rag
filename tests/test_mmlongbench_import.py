import hashlib
import json
from pathlib import Path

import pymupdf
import pytest

from scripts.download_mmlongbench import download_pdf, valid_download
from scripts.prepare_mmlongbench import _as_pages, convert_from_json, normalize_item
from src.eval.dataset_loader import load_eval_dataset, validate_items
from src.eval.metrics import aggregate_metrics, score_prediction

pytestmark = pytest.mark.unit


def raw_question(**overrides):
    return {"doc_id": "report.pdf", "question": "What is the value?", "answer": "9",
            "evidence_pages": "[1, 2]", "evidence_sources": "['Table']",
            "answer_format": "Int", "doc_type": "Financial report", **overrides}


def test_official_strings_and_unanswerable_preserve_original_labels(tmp_path):
    item = normalize_item(raw_question(), 0, tmp_path, page_count=2)
    assert item["gold_pages"] == [1, 2]
    assert Path(item["pdf_path"]).name == "report.pdf"
    assert item["evidence_sources"] == ["Table"]
    assert item["category"] == "table"
    negative = normalize_item(raw_question(answer="Not answerable", answer_format="None"), 1, tmp_path)
    assert negative["answerable"] is False and negative["answer"] == ""
    assert negative["gold_pages"] == []
    assert negative["upstream_evidence_pages"] == "[1, 2]"
    validate_items([item, negative])


@pytest.mark.parametrize("pages,status", [("[]", "missing"), ("[0]", "invalid"),
                                         ("[1, 1418]", "invalid"), ("[true]", "invalid")])
def test_missing_or_invalid_evidence_keeps_qa_and_excludes_page_scores(tmp_path, pages, status):
    item = normalize_item(raw_question(evidence_pages=pages), 0, tmp_path, page_count=2)
    validate_items([item])
    assert item["answerable"] and item["answer"] == "9"
    assert item["evidence_annotation_status"] == status and not item["gold_pages"]
    result = score_prediction({"answer": "9 [p.1]", "retrieved_pages": [{"page_number": 1}]}, item)
    assert result["em"] == 1 and result["citation_f1"] is None
    assert result["page_recall@5"] is None
    good = normalize_item(raw_question(evidence_pages="[1]"), 1, tmp_path, page_count=2)
    scored = score_prediction({"answer": "9 [p.1]", "retrieved_pages": [{"page_number": 1}]}, good)
    summary = aggregate_metrics([result, scored])
    assert summary["answerable_count"] == 2
    assert summary["page_metric_count"] == summary["citation_metric_count"] == 1
    assert summary["citation_f1"] == 1


def test_unmarked_missing_evidence_still_rejected(tmp_path):
    item = normalize_item(raw_question(evidence_pages="[]"), 0, tmp_path)
    item.pop("evidence_annotation_status")
    with pytest.raises(ValueError, match="require an answer and evidence"):
        validate_items([item])
    item["evidence_annotation_status"] = "missing"
    item.pop("evidence_annotation_issue")
    with pytest.raises(ValueError, match="explicit issue"):
        validate_items([item])
    with pytest.raises((ValueError, SyntaxError)):
        _as_pages("__import__('os').getcwd()")


def test_export_preserves_rows_and_relocatable_disjoint_splits(tmp_path):
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    rows = []
    for name in ("first", "second", "third"):
        with pymupdf.open() as pdf:
            pdf.new_page().insert_text((30, 30), name)
            pdf.save(pdfs / f"{name}.pdf")
        rows.extend([raw_question(doc_id=f"{name}.pdf", evidence_pages="[1]"),
                     raw_question(doc_id=f"{name}.pdf", evidence_pages="[]", answer="Not answerable", answer_format="None")])
    rows.append(raw_question(doc_id="first.pdf", evidence_pages="[900]"))
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps(rows), encoding="utf-8")
    output = tmp_path / "eval" / "full.json"
    stats = convert_from_json(raw_path, output, pdfs)
    full = load_eval_dataset(output)
    dev = load_eval_dataset(output.parent / "dev.json")
    test = load_eval_dataset(output.parent / "test.json")
    assert len(full) == len(rows) == stats["full"]["questions"]
    assert {i["id"] for i in dev + test} == {i["id"] for i in full}
    assert not ({i["document_sha256"] for i in dev} & {i["document_sha256"] for i in test})
    assert stats["annotation_issues"] == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert all(not Path(i["pdf_path"]).is_absolute() for i in payload["items"])
    assert {i["id"] for i in load_eval_dataset(output.parent / "smoke.json")} <= {i["id"] for i in dev}


def test_download_cache_checks_content_not_just_file_size(tmp_path):
    path = tmp_path / "doc.pdf"
    original = b"test"
    path.write_bytes(original)
    lfs = {"size": 4, "lfs": {"oid": hashlib.sha256(original).hexdigest()}}
    git = {"size": 4, "oid": hashlib.sha1(b"blob 4\0" + original).hexdigest()}
    assert valid_download(path, lfs) and valid_download(path, git)
    assert valid_download(path, {**lfs, "size": 999})  # stale upstream size, correct content hash
    path.write_bytes(b"bad!")
    assert not valid_download(path, lfs) and not valid_download(path, git)


def test_mirror_fallback_verifies_its_own_hash_and_records_source(tmp_path, monkeypatch):
    original = b"official mirror PDF"
    destination = tmp_path / "doc.pdf"
    part = destination.with_suffix(".pdf.part")
    part.write_bytes(original)
    entry = {"path": "documents/doc.pdf", "size": 123, "lfs": {"oid": "0" * 64}}
    git_path = "data/documents/doc.pdf"
    blob_hash = hashlib.sha1(f"blob {len(original)}\0".encode() + original).hexdigest()
    mirror = {git_path: {"size": len(original), "sha": blob_hash, "path": git_path}}
    calls = []

    def fail_primary(*args, **kwargs):
        calls.append(args)
        raise ValueError("Inconsistent upstream response")

    monkeypatch.setattr("scripts.download_mmlongbench.download_file", fail_primary)
    source = download_pdf(entry, destination, "hf-revision", mirror)
    assert source["provider"] == "github" and source["git_blob_sha1"] == blob_hash
    assert destination.read_bytes() == original and not part.exists()
    assert download_pdf(entry, destination, "hf-revision", mirror) == source
    assert len(calls) == 1
