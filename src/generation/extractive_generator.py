"""Lightweight extractive generator for CPU demo (no VLM required).

Builds a cited answer from BM25 page snippets when lexical overlap is strong;
otherwise returns UNANSWERABLE so refusal/grounding paths can be exercised.
"""

from __future__ import annotations

import re

from src.retrieval.bm25_retriever import tokenize


def _sentences(text: str):
    parts = re.split(r"(?<=[.!?。；;])\s+|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]


def _overlap_score(question_tokens, sentence_tokens):
    if not question_tokens or not sentence_tokens:
        return 0.0
    q = set(question_tokens)
    s = set(sentence_tokens)
    return len(q & s) / max(len(q), 1)


def answer_extractive_with_citations(question, retrieved_hits, min_overlap=0.2):
    """Return a grounded extractive answer or UNANSWERABLE.

    retrieved_hits: list with page_number/page_index and text_snippet.
    """
    q_tokens = tokenize(question)
    # Drop ultra-common question words for scoring
    stop = {"what", "which", "who", "how", "why", "when", "where", "is", "are", "the", "a", "an", "of", "did", "does", "do", "this", "that", "in", "on", "for", "to"}
    q_tokens = [t for t in q_tokens if t not in stop]

    candidates = []
    for hit in retrieved_hits:
        page = hit.get("page_number") or (hit.get("page_index", 0) + 1)
        snippet = hit.get("text_snippet") or hit.get("text") or ""
        for sent in _sentences(snippet):
            score = _overlap_score(q_tokens, tokenize(sent))
            # Boost if numeric tokens from question appear
            if any(t for t in q_tokens if any(ch.isdigit() for ch in t)) and any(
                ch.isdigit() for ch in sent
            ):
                score += 0.15
            candidates.append((score, page, sent))

    candidates.sort(key=lambda x: x[0], reverse=True)
    if not candidates or candidates[0][0] < min_overlap:
        return "UNANSWERABLE"

    best_score, best_page, best_sent = candidates[0]

    # Prefer a compact answer span: if a short clause/number is highly overlapping, use it.
    compact = _compact_span(best_sent, q_tokens)
    claim = compact or best_sent.rstrip(".")
    return "{0} [p.{1}]".format(claim, best_page)


def _compact_span(sentence, question_tokens):
    """Extract a shorter answer-like span when question asks for a specific fact."""
    # Prefer percentages / decimals / bracketed ranges mentioned in the sentence
    import re

    numeric = re.findall(r"\[[^\]]+\]|\d+(?:\.\d+)?%?", sentence)
    if numeric and any(any(ch.isdigit() for ch in t) or t in {"accuracy", "recall", "weight", "range"} for t in question_tokens):
        # Keep distinctive numeric answers short
        for cand in numeric:
            if "%" in cand or cand.startswith("[") or "." in cand:
                return cand

    # Named model / component style answers
    for pattern in (
        r"\bQwen2-VL\b",
        r"\bColPali\b",
        r"\bBM25\b",
        r"page-level grounding",
        r"lose layout",
        r"YES/NO whether each cited page supports the claim",
    ):
        m = re.search(pattern, sentence, flags=re.IGNORECASE)
        if m and any(t in pattern.lower() or t in m.group(0).lower() for t in question_tokens):
            return m.group(0)

    # Fall back to sentence if short enough
    if len(sentence.split()) <= 12:
        return sentence.rstrip(".")
    return None
