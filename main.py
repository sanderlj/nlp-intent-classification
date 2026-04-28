"""Entry point for the CSC5035Z Assignment 1 project scaffold."""

from __future__ import annotations

from src.config import LANGUAGES, RAW_DATA_DIR, ensure_directories
from src.data_utils import load_language_dataset, summarize_dataset


def main() -> None:
    ensure_directories()
    print("Assignment 1 project scaffold is ready.")
    print(f"Configured languages: {LANGUAGES}")
    print(f"Expected raw data directory: {RAW_DATA_DIR}")

    for language in LANGUAGES:
        try:
            dataset = load_language_dataset(language, RAW_DATA_DIR)
        except FileNotFoundError:
            print(f"[pending] No dataset files found yet for language '{language}'.")
            continue

        summary = summarize_dataset(dataset)
        print(f"[loaded] {language}: {summary}")


if __name__ == "__main__":
    main()
