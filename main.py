"""Entry point for the CSC5035Z Assignment 1 project scaffold."""

from __future__ import annotations

from src.config import (
    BPE_MERGE_VALUES,
    CHAR_NGRAM_RANGES,
    LANGUAGES,
    LR_C_VALUES,
    RANDOM_SEED,
    RAW_DATA_DIR,
    ensure_directories,
)
from bpe import BPETokenizer
from src.data_utils import LanguageDataset, load_language_dataset, summarize_dataset
from src.experiments import run_part1_baselines


def run_part2_bpe(language_datasets: dict[str, LanguageDataset]) -> None:
    """Part 2: BPE training and tokenization analysis."""
    
    for language, dataset in language_datasets.items():
        print(f"\n[Part 2] Processing language: {language}")
        
        # Build word list from training texts.
        train_texts = dataset.train.texts
        word_list = []
        for text in train_texts:
            word_list.extend(text.split())
        
        print(f"  Unique words in training set: {len(set(word_list))}")
        
        # Train separate BPETokenizer for each k in BPE_MERGE_VALUES.
        tokenizers = {}
        for k in BPE_MERGE_VALUES:
            tokenizer = BPETokenizer()
            tokenizer.train(word_list, num_merges=k)
            tokenizers[k] = tokenizer
            print(f"  Trained BPE tokenizer with {k} merges.")
            
        # Analyze tokenizations for a few example utterances at k = 100, 300, 500.
        analysis_k_values = [100, 300, 500]  
        
        for k in analysis_k_values:
            print(f"  Example tokenization with k={k}:")
            tokenizer = tokenizers[k]
            for i, text in enumerate(train_texts[:5]):
                tokens = tokenizer.encode(text)
                print(f"    Utterance {i}: {tokens}")
        
        
        # Optionally save these outputs to outputs/tokenization_examples/
        # so they can be copied into the report.
        #
        # If this function will be reused in Part 3/4, return tokenizers or
        # a mapping {language: {k: tokenizer}} instead of printing only.


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

        print("\nRunning Part 2 BPE...")
        run_part2_bpe(language_datasets)


if __name__ == "__main__":
    main()
