"""Aggregate ablation / benchmark JSON results into a comparison table.

Usage:
  python scripts/report_results.py --input-dir results/ablation
  python scripts/report_results.py --input results/benchmark.json --markdown results/report.md
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

METRIC_KEYS = [
    "count",
    "em",
    "f1",
    "page_recall@k",
    "page_precision@k",
    "citation_f1",
    "grounding_verified_rate",
    "refusal_accuracy",
    "refusal_precision",
    "refusal_recall",
    "refusal_f1",
]


def _load_summary(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "summary" in data:
        return data.get("method") or Path(path).stem, data["summary"]
    # ablation_summary.json style: {name: summary}
    if all(isinstance(v, dict) for v in data.values()):
        return None, data
    return Path(path).stem, data


def collect_summaries(input_path=None, input_dir=None):
    rows = {}
    if input_path:
        name, payload = _load_summary(input_path)
        if name is None:
            rows.update(payload)
        else:
            rows[name] = payload
    if input_dir:
        input_dir = Path(input_dir)
        summary_file = input_dir / "ablation_summary.json"
        if summary_file.exists():
            _, payload = _load_summary(summary_file)
            rows.update(payload)
        else:
            for path in sorted(input_dir.glob("*.json")):
                name, payload = _load_summary(path)
                if name is None:
                    rows.update(payload)
                else:
                    rows[name] = payload
    return rows


def to_markdown(rows):
    headers = ["method"] + METRIC_KEYS
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for method, summary in rows.items():
        cells = [method]
        for key in METRIC_KEYS:
            val = summary.get(key)
            if val is None:
                cells.append("-")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def to_csv(rows):
    headers = ["method"] + METRIC_KEYS
    lines = [",".join(headers)]
    for method, summary in rows.items():
        cells = [method]
        for key in METRIC_KEYS:
            val = summary.get(key)
            cells.append("" if val is None else str(val))
        lines.append(",".join(cells))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Build experiment comparison tables")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--input-dir", type=str, default="results/ablation")
    parser.add_argument("--markdown", type=str, default="results/ablation_report.md")
    parser.add_argument("--csv", type=str, default="results/ablation_report.csv")
    args = parser.parse_args()

    rows = collect_summaries(args.input, args.input_dir)
    if not rows:
        raise SystemExit("No result summaries found.")

    md = to_markdown(rows)
    csv_text = to_csv(rows)

    if args.markdown:
        out = Path(args.markdown)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print("Wrote", out)
    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(csv_text, encoding="utf-8")
        print("Wrote", out)

    print(md)


if __name__ == "__main__":
    main()
