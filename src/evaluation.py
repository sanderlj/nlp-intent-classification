"""Evaluation helpers for reporting assignment results."""

from __future__ import annotations

from collections import Counter
from itertools import combinations

from sklearn.metrics import confusion_matrix


def top_confused_pairs(y_true: list[str], y_pred: list[str], top_k: int = 3) -> list[tuple[tuple[str, str], int]]:
    labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    pair_counts: Counter[tuple[str, str]] = Counter()

    for i, j in combinations(range(len(labels)), 2):
        confusion_total = int(matrix[i, j] + matrix[j, i])
        if confusion_total:
            pair_counts[(labels[i], labels[j])] = confusion_total

    return pair_counts.most_common(top_k)
