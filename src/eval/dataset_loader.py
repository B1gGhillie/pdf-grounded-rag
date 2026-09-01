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
    return [resolve_pdf_path(item, dataset_dir, require_exists=require_pdf) for item in items]


def load_mmlongbench_subset(path, limit=None, require_pdf=True):
    """Load a local JSON export of MMLongBench-Doc questions.

    Expected fields per item: doc_id, pdf_path, question, answer, answerable, gold_pages.
    """
    items = load_eval_dataset(path, require_pdf=require_pdf)
    if limit is not None:
        items = items[:limit]
    return items
