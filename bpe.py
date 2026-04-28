"""Standalone Byte-Pair Encoding tokenizer for Assignment 1."""

from __future__ import annotations

from collections import Counter


class BPETokenizer:
    """Simple BPE tokenizer with train and encode methods.

    This is a scaffold implementation. The core learning and merge logic still
    needs to be filled in for the assignment.
    """

    def __init__(self, end_of_word: str = "_") -> None:
        self.end_of_word = end_of_word
        self.merges: list[tuple[str, str]] = []

    def train(self, words: list[str], num_merges: int) -> None:
        """Learn BPE merges from a training corpus."""
        raise NotImplementedError("Implement BPE merge training here.")

    def encode_word(self, word: str) -> list[str]:
        """Encode a single word using learned merges."""
        raise NotImplementedError("Implement word encoding here.")

    def encode(self, text: str) -> list[str]:
        """Encode an utterance by applying BPE word by word."""
        tokens: list[str] = []
        for word in text.strip().split():
            tokens.extend(self.encode_word(word))
        return tokens


def get_pair_frequencies(vocabulary: Counter[tuple[str, ...]]) -> Counter[tuple[str, str]]:
    """Count adjacent symbol-pair frequencies in a BPE vocabulary."""
    pair_counts: Counter[tuple[str, str]] = Counter()
    for symbols, freq in vocabulary.items():
        for index in range(len(symbols) - 1):
            pair_counts[(symbols[index], symbols[index + 1])] += freq
    return pair_counts
