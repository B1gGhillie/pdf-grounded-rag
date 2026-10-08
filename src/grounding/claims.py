"""Citation-attached claim units and conservative textual evidence checks.

These checks catch simple contradictions; they are not semantic entailment.
"""
import re

from src.utils.text import content_tokens, sentences
from src.grounding.citation import extract_cited_pages, strip_citations

NEGATION = re.compile(r"未|没有|不(?:是|含|超过|低于|得|提供|支持|包含|适用|存在)|禁止|无(?:需|法|数据)|\b(?:not|no|never|without)\b", re.I)
ABSENCE = re.compile(r"未(?:披露|提供|报告|说明|给出)|没有(?:提供|说明|数据)|不(?:包含|提供)(?:相关)?(?:信息|数据|细节)|omitted|not (?:reported|provided|discussed)|does not discuss", re.I)


def split_claims(answer):
    """A consecutive citation group supports the preceding sentence(s).

    Uncited trailing sentences stay uncited. This is a deterministic surface
    segmentation, not a claim of full linguistic/atomic decomposition.
    """
    result, cursor = [], 0
    pattern = re.compile(r"(?:\[p\.\d+\][ \t]*)+", re.I)
    for match in pattern.finditer(answer):
        text = answer[cursor:match.start()].strip(" \n\t。.!?；;")
        pages = extract_cited_pages(match.group())
        if text:
            for sent in sentences(text):
                result.append({"claim": sent, "pages": pages})
        cursor = match.end()
    for sent in sentences(answer[cursor:].strip(" \n\t。.!?；;")):
        result.append({"claim": strip_citations(sent), "pages": []})
    return result


def numeric_values(text):
    # Chinese characters adjacent to a numeral must not suppress the match.
    return set(re.sub(r"\s+", "", m).replace("％", "%") for m in
               re.findall(r"[+-]?\d+(?:\.\d+)?\s*(?:%|％|万元|亿元|元|万吨|吨|毫米|厘米|米|小时|分钟|秒|年|月|天|个|次)?", text))


def check_text_support(claim, evidence, question="", min_overlap=0.6):
    """Inspect a relevant sentence for overlap, numeric units, and negation."""
    claim_set = set(content_tokens(claim))
    query_set = set(content_tokens(question))
    candidates = sentences(evidence)
    if not candidates or not claim_set:
        return {"supported": False, "reason": "empty_evidence", "overlap": 0.0, "evidence": ""}
    def rank(s):
        st = set(content_tokens(s))
        return len(claim_set & st)/len(claim_set) + 0.3*len(query_set & st)/max(len(query_set), 1)
    selected = max(candidates, key=rank)
    overlap = len(claim_set & set(content_tokens(selected))) / len(claim_set)
    reason = "verified"
    if numeric_values(claim) - numeric_values(selected):
        reason = "numeric_mismatch"
    elif ABSENCE.search(selected) and not ABSENCE.search(claim):
        reason = "explicit_absence"
    elif bool(NEGATION.search(claim)) != bool(NEGATION.search(selected)):
        reason = "negation_mismatch"
    elif overlap < min_overlap:
        reason = "insufficient_overlap"
    return {"supported": reason == "verified", "reason": reason,
            "overlap": round(overlap, 4), "evidence": selected}


def verify_claim_units(answer, retrieved_hits, question="", min_overlap=0.6, verifier=None):
    """Every claim must be supported; VLM can evaluate joint multi-page evidence."""
    units = split_claims(answer)
    hits = {h.get("page_number", h.get("page_index", -1)+1): h for h in retrieved_hits}
    results = []
    for unit in units:
        pages = unit["pages"]
        if not pages:
            result = {"supported": False, "reason": "uncited_claim"}
        elif any(p not in hits for p in pages):
            result = {"supported": False, "reason": "citation_outside_retrieval"}
        elif verifier is not None:
            result = verifier(unit["claim"], [hits[p] for p in pages], question)
        else:
            checks = [check_text_support(unit["claim"], hits[p].get("text") or hits[p].get("text_snippet", ""),
                                         question, min_overlap) for p in pages]
            # A claim may have alternative support pages. An unsupported extra
            # citation is recorded separately from claim coverage.
            result = max(checks, key=lambda c: (c["supported"], c["overlap"]))
            result = {**result, "page_checks": [{"page": p, **c} for p, c in zip(pages, checks)]}
        results.append({**unit, **result})
    passed = sum(r["supported"] for r in results)
    failed = sorted({p for r in results if not r["supported"] for p in r["pages"]})
    verified = bool(results) and passed == len(results)
    return {"grounding_verified": verified, "confidence": passed/len(results) if results else 0.0,
            "claim_coverage": passed/len(results) if results else 0.0,
            "claim_results": results, "verification_results": results,
            "verification_reason": "verified" if verified else "verification_failed",
            "failed_citations": failed,
            "feedback": None if verified else ("; ".join(f"{r['claim']}: {r['reason']}" for r in results if not r["supported"]) or "No factual claim found in the answer.")}
