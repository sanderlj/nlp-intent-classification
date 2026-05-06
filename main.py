"""Entry point for the CSC5035Z Assignment 1 project scaffold."""

from __future__ import annotations

from src.config import (
    BPE_MERGE_VALUES,
    CHAR_NGRAM_RANGES,
    LANGUAGES,
    METRICS_DIR,
    NB_ALPHA_VALUES,
    LR_C_VALUES,
    RANDOM_SEED,
    RAW_DATA_DIR,
    TOKENIZATION_DIR,
    ERROR_ANALYSIS_DIR,
    PLOTS_DIR,
    ensure_directories,
)
from bpe import BPETokenizer
from src.data_utils import LanguageDataset, load_language_dataset, summarize_dataset
from src.reporting import save_results_csv, save_tokenization_examples, save_error_analysis_json, save_part3_alpha_curve_plot
from src.experiments import run_part1_baselines, run_part3_nb, run_part4_feature_engineering, run_error_analysis_for_model, collect_part3_alpha_curve


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
        example_rows = []
        
        for i, text in enumerate(train_texts[:5]):
            row = {
                "example_id": f"{language}_train_{i}",
                "intent": dataset.train.labels[i],
                "text": text,
            }
            for k in analysis_k_values:
                tokenizer = tokenizers[k]
                row[f"tokens_k{k}"] = tokenizer.encode(text)
            
            example_rows.append(row)
        
        save_tokenization_examples(
            language=language,
            examples=example_rows,
            output_dir=TOKENIZATION_DIR,
        )


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
        
        save_results_csv(results, METRICS_DIR / "part1_baselines.csv")
        
        print("\nSummary:")
        for result in results:
            print(
                f"{result.language} | {result.feature_name} | "
                f"dev={result.dev_accuracy:.4f} | test={result.test_accuracy:.4f} | "
                f"params={result.hyperparameters}"
            )

        print("\nRunning Part 2 BPE...")
        run_part2_bpe(language_datasets)
        
        print("\nRunning Part 3 NB...")
        part3_results = run_part3_nb(
            language_datasets,
            alpha_values=NB_ALPHA_VALUES,
            k_values=BPE_MERGE_VALUES,
            random_seed=RANDOM_SEED
        )
        save_results_csv(part3_results, METRICS_DIR / "part3_nb_results.csv")
        
        curve_rows = collect_part3_alpha_curve(
            language_datasets=language_datasets,
            fixed_k_by_language={"amh": 100, "zul": 750},
            alpha_values=NB_ALPHA_VALUES,
        )
        save_part3_alpha_curve_plot(curve_rows, PLOTS_DIR / "part3_alpha_curve.png")
        print("\nRunning Part 4 Feature Engineering...")
        part4_results = run_part4_feature_engineering(
            language_datasets,
            k_values=BPE_MERGE_VALUES,
            c_values=LR_C_VALUES,
            random_seed=RANDOM_SEED
        )
        save_results_csv(part4_results, METRICS_DIR / "part4_results.csv")

        print("\nRunning Error Analysis for best models per language...")
        
        amh_error_analysis = run_error_analysis_for_model(
            language="amh",
            dataset=language_datasets["amh"],
            model_kind="part4_bpe_lr",
            config={"k": 100, "C": 0.5, "feature_name": "bpe_counts_plus_bigrams_plus_length"},
            random_seed=RANDOM_SEED,
        )
        
        zul_error_analysis = run_error_analysis_for_model(
            language="zul",
            dataset=language_datasets["zul"],
            model_kind="part1_char_lr",
            config={"ngram_range": (3, 5), "C": 5.0},
            random_seed=RANDOM_SEED,
        )
        save_error_analysis_json("amh", amh_error_analysis, ERROR_ANALYSIS_DIR)
        save_error_analysis_json("zul", zul_error_analysis, ERROR_ANALYSIS_DIR)   
if __name__ == "__main__":
    main()
