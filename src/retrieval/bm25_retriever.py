"""BM25 text retrieval over per-page text chunks."""

from src.utils.text import tokens, content_tokens, evidence_window, text_quality

from rank_bm25 import BM25Okapi
import math


class PositiveBM25(BM25Okapi):
    """Standard log1p IDF prevents negative/zero scores in tiny PDF corpora."""
    def _calc_idf(self, nd):
        self.idf = {word: math.log1p((self.corpus_size-count+0.5)/(count+0.5))
                    for word, count in nd.items()}
        self.average_idf = sum(self.idf.values()) / max(1, len(self.idf))


def tokenize(text):
    """Simple tokenizer for mixed English/Chinese PDF text."""
    return tokens(text)


class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.pages = []

    def build_index(self, page_paths, text_chunks):
        """Build BM25 index aligned with page_paths (0-based page_index)."""
        text_by_page = {c["page"] - 1: c["text"] for c in text_chunks}
        source_by_page = {c["page"] - 1: c.get("text_source", "native") for c in text_chunks}
        self.pages = []
        corpus = []

        for i, path in enumerate(page_paths):
            text = text_by_page.get(i, "")
            self.pages.append(
                {
                    "page_index": i,
                    "text": text,
                    "page_path": path,
                    "text_density": len(text),
                    "text_quality": text_quality(text),
                    "text_source": source_by_page.get(i, "empty"),
                }
            )
            corpus.append(tokenize(text) if text else ["__empty__"])

        if not corpus:
            raise ValueError("PDF contains no indexed pages")
        self.bm25 = PositiveBM25(corpus)
        return self

    def load_state(self, state):
        self.bm25 = state["bm25"]
        self.pages = state["pages"]
        return self

    def dump_state(self):
        return {"bm25": self.bm25, "pages": self.pages}

    def get_text_densities(self):
        return {p["page_index"]: p["text_density"] for p in self.pages}

    def retrieve(self, query, top_k=5):
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Call build_index() first.")

        query_tokens = content_tokens(query)
        scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in ranked:
            if score <= 0:
                break
            if len(results) >= top_k:
                break
            p = self.pages[idx]
            results.append(
                {
                    "page_index": p["page_index"],
                    "page_path": p["page_path"],
                    "score": float(score),
                    "text_snippet": evidence_window(p["text"], query),
                    "text": p["text"],
                    "query_coverage": len(set(query_tokens) & set(tokenize(p["text"]))) / max(len(set(query_tokens)), 1),
                    "text_quality": p.get("text_quality", text_quality(p["text"])),
                    "text_source": p.get("text_source", "native"),
                    "text_density": p["text_density"],
                }
            )
        return results
