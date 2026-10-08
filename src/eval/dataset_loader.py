"""Load evaluation datasets (JSON / MMLongBench-style)."""

import json
from pathlib import Path


def load_json_dataset(path):
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "items" in data:
        return data["items"]
    if isinstance(data, list):
        return data
    raise ValueError('Dataset must be a list or {"items": [...]} object')


def resolve_pdf_path(item, dataset_dir, require_exists=True):
    pdf_path = item.get("pdf_path")
    if not pdf_path:
        raise ValueError("Dataset item {0} missing pdf_path".format(item.get("id")))
    pdf_path = Path(pdf_path)
    if not pdf_path.is_absolute():
        pdf_path = (Path(dataset_dir) / pdf_path).resolve()
    if require_exists and not pdf_path.exists():
        raise FileNotFoundError("PDF not found: {0}".format(pdf_path))
    item = dict(item)
    item["pdf_path"] = str(pdf_path)
    # Normalize optional fields
    item.setdefault("answerable", True)
    item.setdefault("gold_pages", [])
    item.setdefault("answer", "")
    return item


def load_eval_dataset(path, require_pdf=True):
    path = Path(path)
    items = load_json_dataset(path)
    dataset_dir = path.parent
    resolved = [resolve_pdf_path(item, dataset_dir, require_exists=require_pdf) for item in items]
    validate_items(resolved)
    return resolved


def validate_items(items):
    if not items:
        raise ValueError("Dataset is empty")
    ids = set()
    for item in items:
        ident = item.get("id")
        if not isinstance(ident, str) or not ident or ident in ids:
            raise ValueError(f"Missing/duplicate string id: {ident}")
        ids.add(ident)
        if not isinstance(item.get("question"), str) or not item["question"].strip():
            raise ValueError(f"{ident}: question must be nonempty")
        if type(item.get("answerable")) is not bool:
            raise ValueError(f"{ident}: answerable must be a JSON boolean")
        pages = item.get("gold_pages")
        if not isinstance(pages, list) or any(type(p) is not int or p < 1 for p in pages) or len(pages) != len(set(pages)):
            raise ValueError(f"{ident}: gold_pages must be unique positive 1-based integers")
        if not isinstance(item.get("answer"), str):
            raise ValueError(f"{ident}: answer must be a string")
        if item["answerable"] and (not item["answer"].strip() or not pages):
            raise ValueError(f"{ident}: answerable items require an answer and evidence pages")
        if not item["answerable"] and item["answer"].strip():
            raise ValueError(f"{ident}: unanswerable items must have an empty answer")


def audit_documents(items, max_pages=None):
    """Hash documents, reject invalid page labels and silent truncation."""
    import pymupdf
    from src.eval.provenance import file_sha256
    docs = {}
    for item in items:
        path = item["pdf_path"]
        if path not in docs:
            with pymupdf.open(path) as doc:
                count = len(doc)
            if max_pages is not None and count > max_pages:
                raise ValueError(f"PDF has {count} pages but max_pages_per_doc={max_pages}: {path}. Set null to evaluate the whole document.")
            docs[path] = {"sha256": file_sha256(path), "pages": count}
        if any(p > docs[path]["pages"] for p in item["gold_pages"]):
            raise ValueError(f"{item['id']}: evidence page exceeds PDF length")
        item["document_sha256"] = docs[path]["sha256"]
    return docs


def split_by_document(items, dev_fraction=0.3, seed=42):
    """Content hashes prevent renamed copies leaking between dev and test."""
    import random
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must lie strictly between 0 and 1")
    audit_documents(items)
    hashes = sorted({i["document_sha256"] for i in items})
    if len(hashes) < 2:
        raise ValueError("At least two distinct PDFs are required for a document-disjoint split")
    random.Random(seed).shuffle(hashes)
    n = min(len(hashes)-1, max(1, round(len(hashes)*dev_fraction)))
    dev = set(hashes[:n])
    return {"dev": [i for i in items if i["document_sha256"] in dev],
            "test": [i for i in items if i["document_sha256"] not in dev]}


def load_mmlongbench_subset(path, limit=None, require_pdf=True):
    """Load a local JSON export of MMLongBench-Doc questions.

    Expected fields per item: doc_id, pdf_path, question, answer, answerable, gold_pages.
    """
    items = load_eval_dataset(path, require_pdf=require_pdf)
    if limit is not None:
        items = items[:limit]
    return items
