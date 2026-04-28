"""Feature extraction helpers for baseline and BPE-based models."""

from __future__ import annotations

import re
import string
from collections import Counter


PUNCT_TRANSLATION = str.maketrans("", "", string.punctuation)


def normalize_for_word_baseline(text: str) -> str:
    """Lowercase and remove ASCII punctuation for the unigram baseline."""
    return text.lower().translate(PUNCT_TRANSLATION)


def whitespace_tokenize(text: str) -> list[str]:
    return [token for token in normalize_for_word_baseline(text).split() if token]


def char_ngrams(text: str, min_n: int, max_n: int) -> list[str]:
    text = re.sub(r"\s+", " ", text.lower()).strip()
    ngrams: list[str] = []
    for n in range(min_n, max_n + 1):
        if len(text) < n:
            continue
        ngrams.extend(text[i : i + n] for i in range(len(text) - n + 1))
    return ngrams


def token_counts(tokens: list[str]) -> Counter:
    return Counter(tokens)
