"""Build weakly evidence-grounded Chinese SFT examples from downloaded FinGLM PDFs.

The upstream FinGLM answers do not contain evidence pages. This script therefore
uses exact numeric/string matches in the downloaded PDFs only as weak training
evidence. It never writes these pages into an evaluation gold file.
"""
from __future__ import annotations
import argparse, json, re, unicodedata
from collections import Counter
from pathlib import Path
import pymupdf as fitz

NUM = re.compile(r"(?<![\d.])-?\d[\d,]*(?:\.\d+)?%?(?![\d.])")

def norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "").replace(" ", "").replace("，", ",")

def numbers(s: str) -> list[str]:
    return [x.replace(",", "") for x in NUM.findall(norm(s)) if len(x.replace(",", "")) >= 2]

def terms(row: dict) -> list[str]:
    p = row.get("prompt") or {}
    out = [p.get("ent_short_name", ""), p.get("year", ""), p.get("key_word", "")]
    out.extend(k for k in p if len(k) >= 3 and k not in {"ent_name", "ent_short_name", "prom_answer", "key_word"})
    return [norm(x) for x in out if x]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/finglm_cn"))
    ap.add_argument("--max-examples", type=int, default=400)
    ap.add_argument("--max-pages", type=int, default=2)
    args = ap.parse_args()
    root = args.root.resolve()
    rows = [json.loads(x) for x in (root / "raw" / "pre.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = [r for r in rows if int(r["id"]) % 10 != 0]  # same held-out rule as prepare_finglm_sft.py
    docs = []
    for pdf in sorted((root / "pdfs").glob("*.pdf")):
        doc = fitz.open(pdf)
        pages = [norm(page.get_text("text")) for page in doc]
        docs.append((pdf, pages))

    out_rows, counts = [], Counter()
    for row in rows:
        entity = norm((row.get("prompt") or {}).get("ent_name", ""))
        year = str((row.get("prompt") or {}).get("year", ""))
        answer = (row.get("answer") or [""])[0]
        ans_nums = numbers(answer)
        best = None
        for pdf, pages in docs:
            if entity and entity not in norm(pdf.name):
                continue
            for page_no, text in enumerate(pages, 1):
                score = 0
                matched_nums = sum(1 for n in ans_nums if n in text)
                score += matched_nums * 5
                score += sum(1 for t in terms(row) if len(t) >= 2 and t in text)
                if year and year in text:
                    score += 1
                if score and (best is None or score > best[0]):
                    best = (score, pdf, page_no, text, matched_nums)
        if best is None or (ans_nums and best[4] == 0):
            counts["unmatched"] += 1
            continue
        score, pdf, page_no, text, matched_nums = best
        # Add a second nearby/high-scoring page when available.
        candidates = []
        pages = next(pages for p, pages in docs if p == pdf)
        for no, page_text in enumerate(pages, 1):
            if no == page_no:
                continue
            s = sum(1 for n in ans_nums if n in page_text) * 5 + sum(1 for t in terms(row) if len(t) >= 2 and t in page_text)
            if s:
                candidates.append((s, no, page_text))
        candidates.sort(reverse=True)
        selected = [(page_no, text)] + [(no, page_text) for _, no, page_text in candidates[: max(0, args.max_pages - 1)]]
        context = "\n\n".join(f"[第{no}页]\n{text[:1800]}" for no, text in selected)
        out_rows.append({
            "id": f"finglm_grounded_{int(row['id']):05d}", "source_id": row["id"],
            "question": row["question"], "answer": answer, "type": row.get("type"),
            "entity": entity, "year": year, "context": context,
            "evidence_pages": [no for no, _ in selected],
            "evidence_status": "weak_exact_match", "evidence_pdf": pdf.name,
        })
        counts["matched"] += 1
        counts[pdf.name] += 1
        if len(out_rows) >= args.max_examples:
            break
    out = root / "sft"; out.mkdir(parents=True, exist_ok=True)
    target = out / "grounded_train.jsonl"
    with target.open("w", encoding="utf-8") as f:
        for item in out_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    report = {"examples": len(out_rows), "counts": dict(counts), "max_pages": args.max_pages,
              "label_type": "weak_exact_match_training_only", "gold_pages": False}
    (out / "grounded_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))

if __name__ == "__main__":
    main()
