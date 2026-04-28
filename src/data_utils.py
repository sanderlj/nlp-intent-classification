"""Utilities for loading and inspecting the INJONGO dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class DatasetSplit:
    texts: list[str]
    labels: list[str]


@dataclass
class LanguageDataset:
    train: DatasetSplit
    dev: DatasetSplit
    test: DatasetSplit


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _split_from_records(records: Iterable[dict], text_key: str, label_key: str) -> DatasetSplit:
    texts = []
    labels = []
    for row in records:
        texts.append(str(row[text_key]))
        labels.append(str(row[label_key]))
    return DatasetSplit(texts=texts, labels=labels)


def load_language_dataset(
    language_code: str,
    data_root: Path,
    text_key: str = "text",
    label_key: str = "intent",
) -> LanguageDataset:
    """Load a language dataset from jsonl files.

    Expected layout:
    data_root/<language_code>/train.jsonl
    data_root/<language_code>/dev.jsonl
    data_root/<language_code>/test.jsonl
    """
    language_dir = data_root / language_code
    train_records = _read_jsonl(language_dir / "train.jsonl")
    dev_records = _read_jsonl(language_dir / "dev.jsonl")
    test_records = _read_jsonl(language_dir / "test.jsonl")

    return LanguageDataset(
        train=_split_from_records(train_records, text_key, label_key),
        dev=_split_from_records(dev_records, text_key, label_key),
        test=_split_from_records(test_records, text_key, label_key),
    )


def summarize_dataset(dataset: LanguageDataset) -> dict[str, int]:
    """Return split sizes for quick inspection."""
    return {
        "train_size": len(dataset.train.texts),
        "dev_size": len(dataset.dev.texts),
        "test_size": len(dataset.test.texts),
        "num_labels_train": len(set(dataset.train.labels)),
    }
