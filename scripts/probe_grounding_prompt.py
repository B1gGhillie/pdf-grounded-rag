"""Replay selected DEV predictions with an alternate grounding prompt.

By default retrieval pages, evidence windows, model, precision, and decoding
stay fixed. --gold-evidence explicitly substitutes annotated pages for diagnosis.
This post-hoc diagnostic is not a benchmark score or an independent validation.
The model is kept loaded; probe timings are not comparable with full RAG timings.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.dataset_loader import load_eval_dataset
from src.eval.metrics import exact_match, token_f1
from src.eval.provenance import atomic_json, collect_provenance, file_sha256
from src.generation.vlm_generator import answer_with_grounding, load_vlm
from src.grounding.citation import extract_cited_pages, page_number_from_hit
from src.grounding.pipeline import apply_grounding
from src.grounding.verifier import verify_claim_on_pages
from src.pipeline.rag_pipeline import _repair_missing_citation
from src.utils.text import evidence_window


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt-language", choices=["en", "zh"], default="en")
    parser.add_argument("--gold-evidence", action="store_true",
                        help="Diagnostic only: give the model annotated evidence pages, never the gold answer")
    args = parser.parse_args()
    if Path(args.output).resolve() in {Path(args.results).resolve(), Path(args.dataset).resolve()}:
        parser.error("Output must not overwrite input files")
    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if dataset.get("split") not in {"dev", "pilot_from_dev", "smoke_from_dev"}:
        parser.error("Only DEV-derived data may be used for prompt development")
    original = json.loads(Path(args.results).read_text(encoding="utf-8"))
    if original["provenance"]["dataset_sha256"] != file_sha256(args.dataset):
        raise ValueError("Source result and dataset hashes differ")
    gold = {i["id"]: i for i in load_eval_dataset(args.dataset)}
    predictions = {p["id"]: p for p in original["predictions"]}
    if len(set(args.ids)) != len(args.ids) or any(i not in predictions or i not in gold for i in args.ids):
        raise ValueError("IDs must be distinct and present in both inputs")
    cfg = copy.deepcopy(original["provenance"]["config"])
    cfg["grounding"]["prompt_language"] = args.prompt_language
    if cfg["runtime"]["generation_backend"] != "vlm":
        raise ValueError("Source results must use a VLM")
    output = {"status": "incomplete", "scope": "post_hoc_dev_gold_evidence_probe" if args.gold_evidence else "post_hoc_dev_prompt_probe",
              "page_selection": "gold_evidence_diagnostic_only" if args.gold_evidence else "original_retrieval",
              "source_results_sha256": file_sha256(args.results), "ids": args.ids,
              "provenance": collect_provenance(cfg, args.dataset, [gold[i] for i in args.ids]), "rows": []}
    atomic_json(args.output, output)
    model, processor = load_vlm(cfg["generation"]["model"],
        load_in_4bit=cfg["generation"].get("load_in_4bit", False),
        device=cfg["runtime"]["device"], revision=cfg["generation"].get("revision"))
    for ident in args.ids:
        previous, item = predictions[ident], gold[ident]
        hits = previous["retrieved_pages"]
        if args.gold_evidence:
            import pymupdf
            if not item["answerable"] or not item["gold_pages"]:
                raise ValueError("Gold-page probes require answerable items with evidence labels")
            if cfg["pdf"].get("text_mode", "native") != "native":
                raise ValueError("Gold-page probe currently supports native text only")
            page_dir = Path(hits[0]["page_path"]).parent
            with pymupdf.open(item["pdf_path"]) as pdf:
                hits = [{"page_number": page, "page_index": page - 1,
                         "page_path": str(page_dir / f"page_{page:04d}.png"),
                         "text": pdf[page-1].get_text("text", sort=True).strip()}
                        for page in item["gold_pages"]]
        page_hits = [h for h in hits if h.get("page_path")]
        if not page_hits or any(not Path(h["page_path"]).is_file() for h in page_hits):
            raise ValueError(f"{ident}: original rendered evidence is missing")
        texts = ([evidence_window(h.get("text", ""), item["question"],
                  cfg["generation"].get("max_context_chars_per_page", 800)) for h in page_hits]
                 if cfg["generation"].get("use_text_evidence", True) else None)
        # Gold answers never enter inference. Gold pages are used only in the
        # explicitly labeled oracle-evidence diagnostic, not a retrieval score.
        answer = answer_with_grounding(model, processor, item["question"],
            [h["page_path"] for h in page_hits], [page_number_from_hit(h) for h in page_hits],
            max_new_tokens=cfg["generation"]["max_new_tokens"],
            evidence_texts=texts, prompt_language=args.prompt_language)
        cited, repaired = _repair_missing_citation(answer, hits, cfg["grounding"])
        grounding = apply_grounding(model, processor, cited, hits, cfg["grounding"], question=item["question"])
        # Recheck old candidates on exactly their old cited pages to isolate the verifier prompt.
        old_claim_checks = []
        old_claims = [] if args.gold_evidence else (previous.get("grounding") or {}).get("claim_results", [])
        for claim in old_claims:
            claim_hits = [h for h in hits if page_number_from_hit(h) in claim["pages"]]
            if claim_hits and len(claim_hits) == len(set(claim["pages"])):
                checked = verify_claim_on_pages(model, processor, claim["claim"], claim_hits,
                                               item["question"], prompt_language=args.prompt_language)
                old_claim_checks.append({"claim": claim["claim"], "pages": claim["pages"],
                    "original_supported": claim["supported"], "alternate_verification": checked})
        row = {"id": ident, "question": item["question"], "gold_answer": item["answer"],
               "gold_answerable": item["answerable"], "original_raw_answer": previous.get("raw_answer"),
               "input_page_numbers": [page_number_from_hit(h) for h in page_hits],
               "original_refusal_reason": previous.get("refusal_reason"), "alternate_raw_answer": answer,
               "alternate_raw_citations": extract_cited_pages(answer), "citation_repaired": repaired,
               "alternate_grounding": grounding, "original_claim_rechecks": old_claim_checks,
               "raw_lexical_em": exact_match(answer, item["answer"]) if item["answerable"] else None,
               "raw_lexical_f1": token_f1(answer, item["answer"]) if item["answerable"] else None}
        output["rows"].append(row)
        atomic_json(args.output, output)
        print(json.dumps({k: row[k] for k in ["id", "alternate_raw_answer", "alternate_raw_citations", "raw_lexical_em"]}, ensure_ascii=True), flush=True)
    output["status"] = "complete"
    atomic_json(args.output, output)


if __name__ == "__main__":
    main()
