"""Download / convert MMLongBench-Doc style evaluation data.

MMLongBench-Doc is a long-document multimodal QA benchmark.
This script supports two modes:

1) --from-json: normalize a local JSON/JSONL export into our eval schema
2) --from-hf:   attempt HuggingFace datasets download (optional dependency)

Expected output item fields:
  id, pdf_path, question, answer, answerable, gold_pages, doc_id (optional)

Usage:
  python scripts/prepare_mmlongbench.py --from-json path/to/raw.json --pdf-root data/pdfs/mmlongbench
  python scripts/prepare_mmlongbench.py --from-hf --limit 50 --pdf-root data/pdfs/mmlongbench
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _as_bool(value, default=True):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("yes", "y", "true", "1", "answerable"):
        return True
    if text in ("no", "n", "false", "0", "unanswerable"):
        return False
    return default


def _as_pages(value):
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]
        pages = []
        for p in parts:
            try:
                pages.append(int(p))
            except ValueError:
                continue
        return pages
    if isinstance(value, list):
        pages = []
        for p in value:
            if isinstance(p, dict):
                p = p.get("page") or p.get("page_number") or p.get("page_idx")
            try:
                pages.append(int(p))
            except (TypeError, ValueError):
                continue
        return pages
    return []


def normalize_item(raw, idx, pdf_root):
    """Map heterogeneous MMLongBench-like fields into our schema."""
    question = raw.get("question") or raw.get("query") or raw.get("q")
    if not question:
        return None

    answer = raw.get("answer")
    if answer is None:
        answer = raw.get("answers") or raw.get("label") or ""
    if isinstance(answer, list):
        answer = answer[0] if answer else ""

    answerable = raw.get("answerable")
    if answerable is None:
        answerable = raw.get("is_answerable", raw.get("answerable_label"))
    answerable = _as_bool(answerable, default=bool(str(answer).strip()))

    gold_pages = _as_pages(
        raw.get("gold_pages")
        or raw.get("evidence_pages")
        or raw.get("page_ids")
        or raw.get("pages")
        or raw.get("doc_page")
    )

    doc_id = raw.get("doc_id") or raw.get("document_id") or raw.get("doc") or "doc"
    pdf_path = raw.get("pdf_path") or raw.get("pdf") or raw.get("document")
    if pdf_path:
        pdf_path = str(pdf_path)
        candidate = Path(pdf_path)
        if not candidate.is_absolute():
            under_root = Path(pdf_root) / pdf_path
            if under_root.exists():
                pdf_path = str(under_root)
            elif (Path(pdf_root) / Path(pdf_path).name).exists():
                pdf_path = str(Path(pdf_root) / Path(pdf_path).name)
    else:
        # Convention: pdf_root/{doc_id}.pdf
        guess = Path(pdf_root) / "{0}.pdf".format(doc_id)
        pdf_path = str(guess)

    item_id = raw.get("id") or raw.get("qid") or "{0}_{1}".format(doc_id, idx)

    return {
        "id": str(item_id),
        "doc_id": str(doc_id),
        "pdf_path": pdf_path,
        "question": question,
        "answer": answer if answerable else "",
        "gold_pages": gold_pages,
        "answerable": answerable,
        "category": raw.get("category") or raw.get("type") or "mmlongbench",
    }


def load_raw_records(path):
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    for key in ("items", "data", "examples", "questions"):
        if key in data and isinstance(data[key], list):
            return data[key]
    raise ValueError("Unrecognized JSON structure in {0}".format(path))


def convert_from_json(raw_path, output_path, pdf_root, limit=None, skip_missing_pdf=False):
    records = load_raw_records(raw_path)
    items = []
    skipped = 0
    for i, raw in enumerate(records):
        if limit is not None and len(items) >= limit:
            break
        item = normalize_item(raw, i, pdf_root)
        if item is None:
            skipped += 1
            continue
        if skip_missing_pdf and not Path(item["pdf_path"]).exists():
            skipped += 1
            continue
        items.append(item)

    payload = {
        "name": "mmlongbench_subset",
        "source": str(raw_path),
        "items": items,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return {
        "output": str(output_path),
        "count": len(items),
        "skipped": skipped,
        "answerable": sum(1 for x in items if x["answerable"]),
        "unanswerable": sum(1 for x in items if not x["answerable"]),
    }


def try_download_hf(dataset_name, split, limit):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Install datasets first: pip install datasets\n{0}".format(exc)
        )

    print("Loading HuggingFace dataset:", dataset_name, "split=", split)
    ds = load_dataset(dataset_name, split=split)
    records = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        records.append(dict(row))
    return records


def main():
    parser = argparse.ArgumentParser(description="Prepare MMLongBench-style eval set")
    parser.add_argument("--from-json", type=str, default=None, help="Local JSON/JSONL export")
    parser.add_argument("--from-hf", action="store_true", help="Download via HuggingFace datasets")
    parser.add_argument(
        "--hf-name",
        type=str,
        default="yubo2333/MMLongBench-Doc",
        help="HF dataset id (may change; override if needed)",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--pdf-root", type=str, default="data/pdfs/mmlongbench")
    parser.add_argument("--output", type=str, default="data/eval/mmlongbench_subset.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-missing-pdf", action="store_true")
    args = parser.parse_args()

    Path(args.pdf_root).mkdir(parents=True, exist_ok=True)

    if args.from_json:
        stats = convert_from_json(
            args.from_json,
            args.output,
            args.pdf_root,
            limit=args.limit,
            skip_missing_pdf=args.skip_missing_pdf,
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return

    if args.from_hf:
        records = try_download_hf(args.hf_name, args.split, args.limit)
        tmp = Path("data/eval/_mmlongbench_hf_raw.json")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        stats = convert_from_json(
            tmp,
            args.output,
            args.pdf_root,
            limit=args.limit,
            skip_missing_pdf=args.skip_missing_pdf,
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        print(
            "Note: place corresponding PDFs under {0} (named by doc_id.pdf) "
            "before running benchmarks.".format(args.pdf_root)
        )
        return

    parser.error("Specify --from-json or --from-hf")


if __name__ == "__main__":
    main()
