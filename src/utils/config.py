"""Shared configuration loader."""

from pathlib import Path

import yaml


def load_config(path="configs/default.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def doc_id_from_pdf(pdf_path):
    return Path(pdf_path).stem
