"""Shared configuration loader."""

from pathlib import Path
import hashlib

import yaml


def load_config(path="configs/default.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def doc_id_from_pdf(pdf_path):
    path = Path(pdf_path).resolve()
    return path.stem + "-" + hashlib.sha256(str(path).encode()).hexdigest()[:12]
