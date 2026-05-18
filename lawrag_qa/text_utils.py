"""Text normalization and lightweight Chinese-aware tokenization."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_NUM_RE = re.compile(r"[A-Za-z0-9_]+")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Tokenize Chinese legal text without external dependencies.

    The tokenizer combines Latin/number words, individual CJK characters, and
    CJK bi-grams. It is deliberately simple but stable enough for the MVP's
    BM25 and cosine retrieval.
    """

    text = normalize_text(text).lower()
    tokens: list[str] = []
    tokens.extend(_LATIN_NUM_RE.findall(text))

    cjk_chars = _CJK_RE.findall(text)
    tokens.extend(cjk_chars)
    tokens.extend("".join(pair) for pair in zip(cjk_chars, cjk_chars[1:]))

    legal_terms = [
        "劳动合同",
        "未签",
        "书面劳动合同",
        "二倍工资",
        "经济补偿",
        "违法解除",
        "社会保险",
        "仲裁时效",
        "试用期",
    ]
    for term in legal_terms:
        if term in text:
            tokens.append(term)
    return tokens


def term_counts(text: str) -> Counter[str]:
    return Counter(tokenize(text))
