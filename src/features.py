"""Feature extraction helpers for baseline and BPE-based models."""

from __future__ import annotations

import re
import string
import numpy as np
from collections import Counter
from typing import Iterable


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


def fit_word_unigram_vocabulary(train_texts: Iterable[str]) -> dict[str, int]:
    """Build a word-unigram vocabulary from training texts only."""
    vocab: dict[str, int] = {}
    
    for text in train_texts:
        for token in whitespace_tokenize(text):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def transform_word_unigram_counts(texts: Iterable[str], vocabulary: dict[str, int]) -> np.ndarray:
    """Transform texts into word-unigram count vectors using a fixed vocabulary."""
    
    texts = list(texts)
    x = np.zeros((len(texts), len(vocabulary)), dtype=np.int32)
    
    for i, text in enumerate(texts):
        for token in whitespace_tokenize(text):
            if token in vocabulary:
                x[i, vocabulary[token]] += 1
    return x


def fit_char_ngram_vocabulary(
    train_texts: Iterable[str],
    min_n: int,
    max_n: int,
) -> dict[str, int]:
    
    """Build a character n-gram vocabulary from training texts only."""
    vocab: dict[str, int] = {}
    for text in train_texts:
        for ngram in char_ngrams(text, min_n, max_n):
            if ngram not in vocab:
                vocab[ngram] = len(vocab)
    return vocab


def transform_char_ngram_counts(
    texts: Iterable[str],
    vocabulary: dict[str, int],
    min_n: int,
    max_n: int,
) -> np.ndarray:
    """Transform texts into character n-gram count vectors using a fixed vocabulary."""
    texts = list(texts)
    x = np.zeros((len(texts), len(vocabulary)), dtype=np.int32)
    
    for i, text in enumerate(texts):
        for ngram in char_ngrams(text, min_n, max_n):
            if ngram in vocabulary:
                x[i, vocabulary[ngram]] += 1
    return x


def bpe_token_bigrams(tokens: list[str]) -> list[str]:
    """Create adjacent bigram features from a BPE token sequence."""
    
    bigram_tokens = []
    for i in range(len(tokens) - 1):
        bigram_tokens.append(f"{tokens[i]}||{tokens[i+1]}")
        
    return bigram_tokens


def fit_bpe_bigram_vocabulary(train_docs: Iterable[list[str]]) -> dict[str, int]:
    """Build a BPE-bigram vocabulary from tokenized training documents only."""
    
    vocab: dict[str, int] = {}
    for doc in train_docs:
        for bigram in bpe_token_bigrams(doc):
            if bigram not in vocab:
                vocab[bigram] = len(vocab)
    return vocab


def transform_bpe_bigram_counts(
    docs: Iterable[list[str]],
    vocabulary: dict[str, int],
) -> np.ndarray:
    
    """Transform tokenized docs into BPE-bigram count vectors."""
    docs = list(docs)
    x = np.zeros((len(docs), len(vocabulary)), dtype=np.int32)
    
    for i, doc in enumerate(docs):
        for bigram in bpe_token_bigrams(doc):
            if bigram in vocabulary:
                x[i, vocabulary[bigram]] += 1
    return x

def transform_length_structure_features(texts: Iterable[str]) -> np.ndarray:
    """Create simple length/structure features per utterance.

    Suggested columns:
    - number of whitespace tokens
    - number of characters
    - average token length
    """
    ## TODO your code here ##
    raise NotImplementedError


def concatenate_feature_blocks(feature_blocks: list[np.ndarray]) -> np.ndarray:
    """Concatenate multiple feature matrices column-wise."""
    ## TODO your code here ##
    raise NotImplementedError
