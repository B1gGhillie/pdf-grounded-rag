"""BM25 text retrieval over per-page text chunks."""

import re

from rank_bm25 import BM25Okapi


def tokenize(text):
    """Simple tokenizer for mixed English/Chinese PDF text."""
    return re.findall(r"[\w\u4e00-\u9fff]+", text.lower())


class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.pages = []

    def build_index(self, page_paths, text_chunks):
        """Build BM25 index aligned with page_paths (0-based page_index)."""
        text_by_page = {c["page"] - 1: c["text"] for c in text_chunks}
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
                }
            )
            corpus.append(tokenize(text) if text else ["__empty__"])

        self.bm25 = BM25Okapi(corpus)
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

        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
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
                    "text_snippet": p["text"][:300],
                    "text_density": p["text_density"],
                }
            )
        return results
