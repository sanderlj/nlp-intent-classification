"""Inspect INJONGO data before modeling.

This script performs conservative data checks:
- full inspection on train/dev
- shallow inspection on test by default, to avoid peeking at examples

Outputs:
- outputs/metrics/data_inspection_report.md
- outputs/metrics/data_inspection_summary.json
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.config import LANGUAGES, METRICS_DIR, RAW_DATA_DIR, ensure_directories


EXPECTED_SPLIT_SIZES = {
    "train": 2240,
    "dev": 320,
    "test": 640,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def inspect_records(records: list[dict[str, Any]], split_name: str, deep: bool) -> dict[str, Any]:
    keys = sorted(records[0].keys()) if records else []
    texts = [record.get("text", "") for record in records]
    labels = [record.get("intent", "") for record in records]
    empty_text_count = sum(1 for text in texts if not str(text).strip())
    missing_label_count = sum(1 for label in labels if not str(label).strip())
    label_counter = Counter(str(label) for label in labels)

    summary: dict[str, Any] = {
        "num_examples": len(records),
        "expected_examples": EXPECTED_SPLIT_SIZES.get(split_name),
        "matches_expected_size": len(records) == EXPECTED_SPLIT_SIZES.get(split_name),
        "keys": keys,
        "empty_text_count": empty_text_count,
        "missing_label_count": missing_label_count,
        "num_unique_labels": len(label_counter),
        "top_10_labels": label_counter.most_common(10),
    }

    if deep:
        sample_examples = []
        for record in records[:3]:
            sample_examples.append(
                {
                    "example_id": record.get("example_id", ""),
                    "intent": record.get("intent", ""),
                    "text": record.get("text", ""),
                }
            )

        text_lengths = [len(str(text)) for text in texts]
        summary["min_text_length"] = min(text_lengths) if text_lengths else 0
        summary["max_text_length"] = max(text_lengths) if text_lengths else 0
        summary["avg_text_length"] = round(sum(text_lengths) / len(text_lengths), 2) if text_lengths else 0.0
        summary["sample_examples"] = sample_examples

    return summary


def compare_label_sets(language_summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    train_labels = set(language_summary["train"]["label_set"])
    dev_labels = set(language_summary["dev"]["label_set"])
    test_labels = set(language_summary["test"]["label_set"])

    return {
        "dev_missing_from_train": sorted(dev_labels - train_labels),
        "test_missing_from_train": sorted(test_labels - train_labels),
        "labels_in_all_splits": len(train_labels & dev_labels & test_labels),
    }


def build_language_summary(language: str) -> dict[str, Any]:
    language_dir = RAW_DATA_DIR / language
    split_records = {
        split: read_jsonl(language_dir / f"{split}.jsonl")
        for split in ("train", "dev", "test")
    }

    summary: dict[str, Any] = {"language": language, "splits": {}}
    label_sets: dict[str, set[str]] = {}

    for split, records in split_records.items():
        deep = split in {"train", "dev"}
        split_summary = inspect_records(records, split, deep=deep)
        label_set = sorted({str(record.get("intent", "")) for record in records if str(record.get("intent", "")).strip()})
        split_summary["label_set"] = label_set
        label_sets[split] = set(label_set)
        summary["splits"][split] = split_summary

    summary["cross_split_checks"] = {
        "dev_missing_from_train": sorted(label_sets["dev"] - label_sets["train"]),
        "test_missing_from_train": sorted(label_sets["test"] - label_sets["train"]),
        "train_label_count": len(label_sets["train"]),
        "dev_label_count": len(label_sets["dev"]),
        "test_label_count": len(label_sets["test"]),
    }
    return summary


def format_report(all_summaries: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Data Inspection Report")
    lines.append("")
    lines.append("This report performs full sanity checks on train/dev and only shallow checks on test.")
    lines.append("")

    for summary in all_summaries:
        lines.append(f"## Language: {summary['language']}")
        lines.append("")
        for split in ("train", "dev", "test"):
            split_summary = summary["splits"][split]
            lines.append(f"### {split}")
            lines.append(f"- Examples: {split_summary['num_examples']} (expected {split_summary['expected_examples']})")
            lines.append(f"- Size matches brief: {split_summary['matches_expected_size']}")
            lines.append(f"- Keys: {', '.join(split_summary['keys'])}")
            lines.append(f"- Empty texts: {split_summary['empty_text_count']}")
            lines.append(f"- Missing labels: {split_summary['missing_label_count']}")
            lines.append(f"- Unique labels: {split_summary['num_unique_labels']}")
            lines.append("")

            if split in {"train", "dev"}:
                lines.append(f"- Text length min/avg/max: {split_summary['min_text_length']} / {split_summary['avg_text_length']} / {split_summary['max_text_length']}")
                lines.append("- Sample examples:")
                for example in split_summary["sample_examples"]:
                    text_preview = str(example["text"]).replace("\n", " ").strip()
                    if len(text_preview) > 120:
                        text_preview = text_preview[:117] + "..."
                    lines.append(
                        f"  - `{example['example_id']}` | `{example['intent']}` | {text_preview}"
                    )
                lines.append("")

        cross_split = summary["cross_split_checks"]
        lines.append("### Cross-split checks")
        lines.append(f"- Train label count: {cross_split['train_label_count']}")
        lines.append(f"- Dev label count: {cross_split['dev_label_count']}")
        lines.append(f"- Test label count: {cross_split['test_label_count']}")
        lines.append(f"- Dev labels missing from train: {cross_split['dev_missing_from_train']}")
        lines.append(f"- Test labels missing from train: {cross_split['test_missing_from_train']}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    ensure_directories()
    all_summaries = [build_language_summary(language) for language in LANGUAGES]

    report_path = METRICS_DIR / "data_inspection_report.md"
    json_path = METRICS_DIR / "data_inspection_summary.json"

    report_path.write_text(format_report(all_summaries), encoding="utf-8")
    json_path.write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote report to: {report_path}")
    print(f"Wrote summary to: {json_path}")


if __name__ == "__main__":
    main()
