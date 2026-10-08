"""Bilingual extractive regression backend; not a VLM performance surrogate."""
from src.utils.text import content_tokens, sentences, evidence_window
from src.grounding.claims import ABSENCE


def answer_extractive_with_citations(question, retrieved_hits, min_overlap=0.2):
    query = set(content_tokens(question))
    candidates = []
    for hit in retrieved_hits:
        page = hit.get("page_number") or hit.get("page_index", 0)+1
        text = hit.get("text") or hit.get("text_snippet", "")
        for sentence in sentences(evidence_window(text, question)):
            score = len(query & set(content_tokens(sentence))) / max(len(query), 1)
            candidates.append((score, page, sentence))
    candidates.sort(key=lambda row: (-row[0], row[1], row[2]))
    if not candidates or candidates[0][0] < min_overlap:
        return "UNANSWERABLE"
    score, page, sentence = candidates[0]
    if ABSENCE.search(sentence):
        return "UNANSWERABLE"
    return f"{sentence} [p.{page}]"