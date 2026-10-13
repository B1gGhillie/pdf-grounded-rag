"""Export human-reviewed company-disjoint dev/test QA with provenance checks."""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.dataset_loader import audit_documents, validate_items
from src.eval.provenance import atomic_json, file_sha256


def prepare_reviewed_split(review_path, manifest, root, output_dir, partial=False):
    review_path = Path(review_path).resolve()
    payload = json.loads(review_path.read_text(encoding="utf-8"))
    split = payload.get("split")
    if split not in {"dev", "test"}:
        raise ValueError("Reviewed file must identify dev or test")
    if payload.get("manifest_sha256") != file_sha256(root/"manifest.json"):
        raise ValueError("Review belongs to a different document manifest")
    proposals = json.loads((root/"annotations"/f"{split}.json").read_text(encoding="utf-8"))
    original = {r["id"]: r for r in proposals["items"]}
    reviews = payload.get("items", [])
    ids = [r.get("id") for r in reviews]
    if len(ids) != len(set(ids)) or set(ids) != set(original):
        raise ValueError("Review IDs must match the complete original proposal set; mark unwanted questions rejected")
    documents = {d["doc_id"]: d for d in manifest["documents"]}
    accepted, pending, rejected, checked_hashes = [], [], [], {}
    for row in reviews:
        doc = documents.get(row.get("doc_id"))
        if not doc or doc["split"] != split or row.get("split") != split:
            raise ValueError(f"{row['id']}: invalid document/split")
        source = original[row["id"]]
        for field in ("doc_id", "company_group", "document_sha256"):
            if row.get(field) != source.get(field):
                raise ValueError(f"{row['id']}: changed provenance field {field}")
        pdf = (root/doc["pdf_path"]).resolve()
        if pdf not in checked_hashes:
            checked_hashes[pdf] = file_sha256(pdf)
        if row.get("document_sha256") != doc["sha256"] or checked_hashes[pdf] != doc["sha256"]:
            raise ValueError(f"{row['id']}: PDF content changed")
        status = row.get("annotation_status")
        if status in {"pending", "needs_fix"}:
            pending.append(row["id"])
            continue
        if status not in {"approved", "rejected"}:
            raise ValueError(f"{row['id']}: unknown review status")
        if not isinstance(row.get("reviewer"), str) or not row["reviewer"].strip():
            raise ValueError(f"{row['id']}: reviewer's name/identifier is required")
        if status == "rejected":
            if not isinstance(row.get("review_notes"), str) or not row["review_notes"].strip():
                raise ValueError(f"{row['id']}: rejected labels need a reason")
            rejected.append(row["id"])
            continue
        if type(row.get("answerable")) is not bool:
            raise ValueError(f"{row['id']}: answerable must be boolean")
        if row["answerable"] and row.get("source_pages_checked") is not True:
            raise ValueError(f"{row['id']}: confirm evidence pages against the PDF")
        if not row["answerable"] and row.get("full_document_checked") is not True:
            raise ValueError(f"{row['id']}: negative labels require full-document review")
        if not row["answerable"] and row.get("gold_pages"):
            raise ValueError(f"{row['id']}: unanswerable gold_pages must be empty")
        item = {**row, "pdf_path": str(pdf),
                "evidence_annotation_status": "complete" if row["answerable"] else "not_applicable"}
        accepted.append(item)
    if pending and not partial:
        raise ValueError(f"{split}: {len(pending)} labels still pending; complete review or use --partial for diagnosis")
    if not accepted:
        raise ValueError(f"{split}: no human-approved questions to export")
    validate_items(accepted)
    audit_documents(accepted)
    for row in accepted:
        row["pdf_path"] = os.path.relpath(row["pdf_path"], output_dir).replace("\\", "/")
    report = {"split": split, "original_proposals": len(original), "approved": len(accepted),
              "pending": len(pending), "rejected": len(rejected), "rejected_ids": rejected,
              "documents": len({r["document_sha256"] for r in accepted}),
              "categories": dict(Counter(r["category"] for r in accepted)),
              "answerable": sum(r["answerable"] for r in accepted),
              "unanswerable": sum(not r["answerable"] for r in accepted)}
    output = {"name": manifest["name"], "schema_version": 1, "split": split,
              "annotation_status": "human_reviewed", "partial": bool(pending),
              "evaluation_note": "Pilot convenience sample; template repetition and chart limitations apply.",
              "page_numbering": manifest["page_numbering"], "manifest_sha256": payload["manifest_sha256"],
              "review_file_sha256": file_sha256(review_path), "review_summary": report, "items": accepted}
    return output


def export(review_paths, root, output_dir=None, partial=False):
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve() if output_dir else root/"eval"
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    sets, hash_sets, company_sets = {}, {}, {}
    # Check the full original manifest, not only the subset with approved labels.
    for doc in manifest["documents"]:
        if doc["split"] not in {"dev", "test"}:
            continue
        for table, key in ((hash_sets, doc["sha256"]), (company_sets, doc["company_group"])):
            if key in table and table[key] != doc["split"]:
                raise ValueError("Company/PDF leakage in manifest")
            table[key] = doc["split"]
    for path in review_paths:
        output = prepare_reviewed_split(path, manifest, root, output_dir, partial)
        if output["split"] in sets:
            raise ValueError("Only one reviewed file per split is allowed")
        sets[output["split"]] = output
    for split, payload in sets.items():
        target = output_dir/f"{split}.json"
        if target.exists():
            raise ValueError(f"Export already exists: {target}; choose a new --output-dir to preserve its version")
    # Do not write any output until every requested split passed validation.
    for split, payload in sets.items():
        atomic_json(output_dir/f"{split}.json", payload)
    report = {s: p["review_summary"] for s, p in sets.items()}
    atomic_json(output_dir/"review_report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", nargs="+", required=True, help="Downloaded dev.reviewed.json and/or test.reviewed.json")
    parser.add_argument("--root", default="data/enterprise_cn")
    parser.add_argument("--output-dir")
    parser.add_argument("--partial", action="store_true", help="Export only approved items; diagnostic subset, not full test")
    args = parser.parse_args()
    print(json.dumps(export(args.reviews, args.root, args.output_dir, args.partial), ensure_ascii=False, indent=2))
