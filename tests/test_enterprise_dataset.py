import copy
import json

import pymupdf
import pytest

from scripts.build_enterprise_annotations import number_evidence
from scripts.export_enterprise_eval import export
from scripts.prepare_enterprise_corpus import prepare
from src.eval.dataset_loader import load_eval_dataset, validate_items
from src.eval.provenance import atomic_json, file_sha256

pytestmark = pytest.mark.unit


def fixture_corpus(tmp_path):
    root = tmp_path/"corpus"
    documents = []
    for split in ("dev", "test"):
        pdf = root/"pdfs"/f"{split}.pdf"
        pdf.parent.mkdir(parents=True, exist_ok=True)
        with pymupdf.open() as doc:
            doc.new_page().insert_text((40, 40), split+" revenue 125.00")
            doc.save(pdf)
        documents.append({"doc_id": split, "company_group": split, "split": split,
                          "pdf_path": f"pdfs/{split}.pdf", "sha256": file_sha256(pdf)})
    manifest = {"name": "test", "page_numbering": "physical_1_based", "documents": documents}
    atomic_json(root/"manifest.json", manifest)
    paths = []
    for d in documents:
        row = {"id": d["doc_id"]+"_1", "doc_id": d["doc_id"], "split": d["split"],
               "company_group": d["company_group"], "document_sha256": d["sha256"],
               "pdf_path": "../"+d["pdf_path"], "question": "Revenue?", "answer": "125.00元",
               "answerable": True, "gold_pages": [1], "category": "table",
               "annotation_status": "pending", "reviewer": "", "source_pages_checked": False}
        proposal = {"split": d["split"], "manifest_sha256": file_sha256(root/"manifest.json"), "items": [row]}
        atomic_json(root/"annotations"/f"{d['split']}.json", proposal)
        reviewed = copy.deepcopy(proposal)
        reviewed["items"][0].update(annotation_status="approved", reviewer="annotator_1", source_pages_checked=True)
        path = root/"reviewed"/f"{d['split']}.json"
        atomic_json(path, reviewed)
        paths.append(path)
    return root, paths


def test_proposals_cannot_be_evaluated(tmp_path):
    root, _ = fixture_corpus(tmp_path)
    with pytest.raises(ValueError, match="unreviewed"):
        load_eval_dataset(root/"annotations"/"dev.json")


def test_reviewed_export_is_loadable_and_preserves_company_split(tmp_path):
    root, paths = fixture_corpus(tmp_path)
    report = export(paths, root)
    assert report["dev"]["approved"] == report["test"]["approved"] == 1
    dev = load_eval_dataset(root/"eval"/"dev.json")
    test = load_eval_dataset(root/"eval"/"test.json")
    assert {r["document_sha256"] for r in dev}.isdisjoint(r["document_sha256"] for r in test)
    with pytest.raises(ValueError, match="already exists"):
        export(paths, root)


def test_incomplete_review_never_writes_any_split(tmp_path):
    root, paths = fixture_corpus(tmp_path)
    review = json.loads(paths[1].read_text(encoding="utf-8"))
    review["items"][0]["annotation_status"] = "pending"
    atomic_json(paths[1], review)
    with pytest.raises(ValueError, match="pending"):
        export(paths, root)
    assert not (root/"eval"/"dev.json").exists()


def test_negative_review_requires_full_document_check(tmp_path):
    root, paths = fixture_corpus(tmp_path)
    review = json.loads(paths[0].read_text(encoding="utf-8"))
    review["items"][0].update(answerable=False, answer="", gold_pages=[], category="unanswerable")
    atomic_json(paths[0], review)
    with pytest.raises(ValueError, match="full-document"):
        export([paths[0]], root)
    review["items"][0]["full_document_checked"] = True
    atomic_json(paths[0], review)
    export([paths[0]], root)
    assert load_eval_dataset(root/"eval"/"dev.json")[0]["answerable"] is False


def test_modified_pdf_and_ids_are_rejected(tmp_path):
    root, paths = fixture_corpus(tmp_path)
    review = json.loads(paths[0].read_text(encoding="utf-8"))
    review["items"][0]["id"] = "invented"
    atomic_json(paths[0], review)
    with pytest.raises(ValueError, match="IDs"):
        export([paths[0]], root)
    review["items"][0]["id"] = "dev_1"
    atomic_json(paths[0], review)
    pdf = root/"pdfs"/"dev.pdf"
    pdf.write_bytes(pdf.read_bytes()+b"\n% changed")
    with pytest.raises(ValueError, match="content changed"):
        export([paths[0]], root)


def test_corpus_preflight_rejects_hash_leak_before_copying(tmp_path):
    source = tmp_path/"source"
    source.mkdir()
    with pymupdf.open() as doc:
        doc.new_page().insert_text((40, 40), "same source")
        doc.save(source/"one.pdf")
    (source/"two.pdf").write_bytes((source/"one.pdf").read_bytes())
    specs = {"split_policy": "test", "documents": [
        {"doc_id": name, "filename": filename, "company_group": name, "split": split}
        for name, filename, split in [("one", "one.pdf", "dev"), ("two", "two.pdf", "test")]]}
    path = tmp_path/"specs.json"
    atomic_json(path, specs)
    with pytest.raises(ValueError, match="leakage"):
        prepare(source, tmp_path/"out", path)
    assert not (tmp_path/"out").exists()


def test_source_number_check_requires_correct_sign_and_full_token():
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((40, 40), "Revenue 1,250.00    Loss (125.00)")
        assert number_evidence(page, "1250.00", "revenue", 2025, "yuan")["bbox_candidates_pt"]
        assert number_evidence(page, "-125.00", "loss", 2025, "yuan")["value"] == "-125.00"
        with pytest.raises(ValueError, match="absent"):
            number_evidence(page, "125.00", "loss", 2025, "yuan")
        with pytest.raises(ValueError, match="absent"):
            number_evidence(page, "250.00", "revenue", 2025, "yuan")
