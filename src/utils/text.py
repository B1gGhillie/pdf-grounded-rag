"""Deterministic bilingual tokenization and evidence-window selection."""

import re

TOKENIZER_VERSION = "zh-char-bigram-en-v2"
STOP = set("what which who how why when where is are the a an of did does do this that in on for to and with was were from document report".split())
ZH_STOP = set("的了是在与和及该本中为于有请问什么多少如何哪些是否")


def tokens(text, bigrams=True):
    text = join_chinese_wraps(text)
    result = []
    for run in re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]+\d*|\d+(?:\.\d+)?%?", text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", run):
            result.extend(run)
            if bigrams:
                result.extend(run[i:i+2] for i in range(len(run)-1))
        else:
            result.append(run)
    return result


def content_tokens(text):
    return [t for t in tokens(text) if t not in STOP and t not in ZH_STOP]


def sentences(text):
    # Decimal points and page citations are not sentence boundaries.
    text = join_chinese_wraps(text)
    return [s.strip() for s in re.split(r"(?<=[。！？；!?;])\s*|(?<!\d)\.(?=\s|$)\s*|\n+", text) if s.strip()]


def join_chinese_wraps(text):
    """Repair single soft line breaks; keep paragraph gaps and table rows.

    Digits separated by OCR spaces are deliberately NOT merged: they may be
    distinct table values. Preserve source text for checking such ambiguities.
    """
    return re.sub(r"(?<=[\u4e00-\u9fff，、])\r?\n(?=[\u4e00-\u9fff，。！？])", "", text)


def evidence_window(text, query, max_chars=1800):
    """Pick query-bearing sentences from the entire page, preserving source order."""
    if len(text) <= max_chars:
        return text
    query_tokens = set(content_tokens(query))
    parts = sentences(text)
    ranked = sorted(range(len(parts)), key=lambda i: (-len(query_tokens & set(content_tokens(parts[i]))), i))
    chosen, used = [], 0
    for i in ranked:
        if used + len(parts[i]) <= max_chars:
            chosen.append(i)
            used += len(parts[i]) + 1
    if not chosen:
        # A page can be one unbroken line: center around its best matching token.
        positions = [text.lower().find(t) for t in query_tokens if t in text.lower()]
        start = max(0, min(positions, default=0) - max_chars // 4)
        return text[start:start + max_chars]
    return "\n".join(parts[i] for i in sorted(chosen))


def text_quality(text):
    """Observable extraction quality proxy, not OCR confidence or a probability."""
    if not text.strip():
        return 0.0
    useful = sum(c.isalnum() for c in text)
    replacement = text.count("\ufffd")
    return max(0.0, min(1.0, useful / max(len(text), 1) * (1 - replacement / len(text))))
