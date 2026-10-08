"""Audit annotations and split Chinese document QA by PDF content hash."""
import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.dataset_loader import load_eval_dataset, audit_documents, split_by_document
from src.eval.provenance import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", help="Optional: write disjoint dev/test splits")
    parser.add_argument("--dev-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    items = load_eval_dataset(args.dataset)
    docs = audit_documents(items)
    print(json.dumps({"questions": len(items), "unique_documents": len({d['sha256'] for d in docs.values()}),
                      "answerable": sum(i["answerable"] for i in items), "documents": docs}, ensure_ascii=False, indent=2))
    if args.output_dir:
        out = Path(args.output_dir).resolve()
        for name, split in split_by_document(items, args.dev_fraction, args.seed).items():
            for item in split:
                item["pdf_path"] = os.path.relpath(item["pdf_path"], out).replace("\\", "/")
            atomic_json(out/f"{name}.json", {"split": name, "seed": args.seed, "items": split})


if __name__ == "__main__":
    main()
