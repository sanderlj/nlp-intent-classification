"""Entry point for the CSC5035Z Assignment 1 project scaffold."""

from __future__ import annotations

from src.config import (
    CHAR_NGRAM_RANGES,
    LANGUAGES,
    LR_C_VALUES,
    RANDOM_SEED,
    RAW_DATA_DIR,
    ensure_directories,
)
from src.data_utils import load_language_dataset, summarize_dataset
from src.experiments import run_part1_baselines


def main() -> None:
    ensure_directories()
    print(f"Configured languages: {LANGUAGES}")
    print(f"Expected raw data directory: {RAW_DATA_DIR}")

    language_datasets = {}
    for language in LANGUAGES:
        try:
            dataset = load_language_dataset(language, RAW_DATA_DIR)
        except FileNotFoundError:
            print(f"[pending] No dataset files found yet for language '{language}'.")
            continue

        language_datasets[language] = dataset
        summary = summarize_dataset(dataset)
        print(f"[loaded] {language}: {summary}")

    if language_datasets:
        print("\nRunning Part 1 baselines...")
        results = run_part1_baselines(
            language_datasets=language_datasets,
            c_values=LR_C_VALUES,
            ngram_ranges=CHAR_NGRAM_RANGES,
            random_seed=RANDOM_SEED,
        )
        print("\nSummary:")
        for result in results:
            print(
                f"{result.language} | {result.feature_name} | "
                f"dev={result.dev_accuracy:.4f} | test={result.test_accuracy:.4f} | "
                f"params={result.hyperparameters}"
            )


if __name__ == "__main__":
    main()
