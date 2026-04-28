"""Standalone multinomial Naive Bayes classifier for Assignment 1."""

from __future__ import annotations

from collections import Counter, defaultdict
from math import log


class MultinomialNaiveBayes:
    """Naive Bayes classifier over token-count features.

    This scaffold keeps the public interface in place so the rest of the
    project can be wired up while you implement the probability estimates.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.class_log_priors: dict[str, float] = {}
        self.token_log_probs: dict[str, dict[str, float]] = {}
        self.vocabulary: set[str] = set()

    def fit(self, documents: list[list[str]], labels: list[str]) -> None:
        """Estimate class priors and token likelihoods."""
        raise NotImplementedError("Implement NB training here.")

    def predict_one(self, document: list[str]) -> str:
        """Predict the label for one tokenized document."""
        raise NotImplementedError("Implement NB prediction here.")

    def predict(self, documents: list[list[str]]) -> list[str]:
        return [self.predict_one(document) for document in documents]


def count_tokens_by_class(
    documents: list[list[str]],
    labels: list[str],
) -> dict[str, Counter[str]]:
    """Aggregate token counts per class."""
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for document, label in zip(documents, labels):
        counts[label].update(document)
    return counts
