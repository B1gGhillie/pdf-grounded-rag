import copy
import pytest
from src.ingest.pdf_loader import pdf_to_text_chunks
from src.ingest.pipeline import ingest_document
from src.retrieval.index_store import DocumentIndex
from src.utils.config import doc_id_from_pdf

pytestmark = pytest.mark.integration


def test_cache_invalidates_changed_content_models_and_preprocessing(tmp_path):
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"original")
    cfg = {"pdf": {"dpi": 72}, "retrieval": {"model": "model", "model_revision": "a"}}
    old = DocumentIndex.fingerprint(path, cfg)
    cfg["retrieval"]["model_revision"] = "b"
    assert DocumentIndex.fingerprint(path, cfg) != old
    old = DocumentIndex.fingerprint(path, cfg)
    path.write_bytes(b"changed!")
    assert DocumentIndex.fingerprint(path, cfg) != old
    assert doc_id_from_pdf(tmp_path / "a/doc.pdf") != doc_id_from_pdf(tmp_path / "b/doc.pdf")


def test_native_and_ocr_profiles_have_separate_cache_paths(tmp_path):
    import pymupdf
    path = tmp_path / "test.pdf"
    with pymupdf.open() as doc:
        doc.new_page().insert_text((30, 30), "Example report investment is 350 million.")
        doc.save(path)
    cfg = {"pdf": {"dpi": 72, "text_mode": "native"}, "retrieval": {}}
    index = DocumentIndex("doc", tmp_path / "cache")
    paths1, bm1, _, _ = ingest_document(path, cfg, index, visual_backend="none")
    changed = copy.deepcopy(cfg)
    changed["pdf"]["dpi"] = 96
    paths2, _, _, _ = ingest_document(path, changed, index, visual_backend="none")
    assert paths1 != paths2
    paths3, bm3, _, _ = ingest_document(path, cfg, index, visual_backend="none")
    assert paths1 == paths3 and bm1.pages == bm3.pages


def test_missing_ocr_data_fails_instead_of_empty_success(tmp_path):
    import pymupdf
    path = tmp_path / "blank.pdf"
    with pymupdf.open() as doc:
        doc.new_page()
        doc.save(path)
    with pytest.raises(RuntimeError, match="OCR failed on page 1"):
        pdf_to_text_chunks(path, text_mode="ocr", tessdata=str(tmp_path / "absent"))
    assert pdf_to_text_chunks(path, text_mode="native") == []


def test_no_implicit_fifty_page_cap(monkeypatch, tmp_path):
    from src.ingest import pipeline
    observed = []
    pdf = tmp_path / "long.pdf"
    pdf.write_bytes(b"test input")
    def render(path, out, dpi, max_pages):
        observed.append(max_pages)
        return [str(tmp_path/f"p{i}.png") for i in range(60)]
    monkeypatch.setattr(pipeline, "pdf_to_page_images", render)
    monkeypatch.setattr(pipeline, "pdf_to_text_chunks", lambda *a, **k: [{"page": 60, "text": "Tail evidence"}])
    pages, bm25, _, _ = ingest_document(pdf, {"pdf": {"dpi": 72}, "retrieval": {}},
                                       DocumentIndex("long", tmp_path/"cache"), visual_backend="none")
    assert observed == [None] and len(pages) == 60
    assert bm25.retrieve("Tail evidence")[0]["page_index"] == 59
