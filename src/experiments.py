"""Experiment helpers for Assignment 1. 
This module contains code to run the experiments for Part 1 of the assignment.
"""


from __future__ import annotations

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.features import (
    fit_char_ngram_vocabulary,
    fit_word_unigram_vocabulary,
    transform_char_ngram_counts,
    transform_word_unigram_counts,
)

@dataclass
class ExperimentResult:
    language: str
    model_name: str
    feature_name: str
    dev_accuracy: float
    test_accuracy: float
    hyperparameters: dict
    notes: str = ""


def run_word_unigram(
    language: str,
    dataset,
    c_values: list[float],
    random_seed: int,
) -> ExperimentResult:
    """Tune C on dev for word unigram LR, then evaluate once on test."""
    train_texts = dataset.train.texts
    train_labels = dataset.train.labels
    dev_texts = dataset.dev.texts
    dev_labels = dataset.dev.labels
    test_texts = dataset.test.texts
    test_labels = dataset.test.labels
    
    # Fit vocabulary on train and transform texts into count vectors.
    
    vocab = fit_word_unigram_vocabulary(train_texts) 
    x_train = transform_word_unigram_counts(train_texts, vocab)
    x_dev = transform_word_unigram_counts(dev_texts, vocab)
    x_test = transform_word_unigram_counts(test_texts, vocab)
    
    # Initialize tracking variables for best model selection.
    best_dev_acc = -1.0
    best_c = None
    best_model = None
    
    # Tune C on dev set.
    for c in c_values:
        model = LogisticRegression(
            C=c,
            solver="lbfgs",
            max_iter=2000,
        )
        
        model.fit(x_train, train_labels)
        dev_pred = model.predict(x_dev)
        dev_acc = accuracy_score(dev_labels, dev_pred)
        print(f" [dev] {language} |word_unigram | C={c}: dev acc={dev_acc:.4f}")
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_c = c
            best_model = model
    
    if best_model is None:
        raise ValueError("No model was trained. Check if c_values is empty.")
        
    # Evaluate best model on test set.
    test_pred = best_model.predict(x_test)
    test_acc = accuracy_score(test_labels, test_pred)
    
    # Return experiment result.
    return ExperimentResult(
        language=language,
        model_name="LR",
        feature_name="word_unigram",
        dev_accuracy=float(best_dev_acc),
        test_accuracy=float(test_acc),
        hyperparameters={"C": best_c},
    )


def run_char_ngram(
    language: str,
    dataset,
    ngram_ranges: list[tuple[int, int]],
    c_values: list[float],
    random_seed: int,
) -> ExperimentResult:
    """Tune n-gram range and C on dev for char n-gram LR, then test once."""
    train_texts = dataset.train.texts
    train_labels = dataset.train.labels
    dev_texts = dataset.dev.texts
    dev_labels = dataset.dev.labels
    test_texts = dataset.test.texts
    test_labels = dataset.test.labels
    
    # Track the best configuration across all (ngram_range, C) combinations.
    best_dev_acc = -1.0
    best_c = None
    best_ngram_range = None
    best_model = None
    selected_x_test = None

    # Fit vocabulary on train and transform texts for each n-gram range, then tune C.
    for (min_n, max_n) in ngram_ranges:
        vocab = fit_char_ngram_vocabulary(train_texts, min_n, max_n)
        x_train = transform_char_ngram_counts(train_texts, vocab, min_n, max_n)
        x_dev = transform_char_ngram_counts(dev_texts, vocab, min_n, max_n)
        x_test = transform_char_ngram_counts(test_texts, vocab, min_n, max_n)

        for c in c_values:
            model = LogisticRegression(
                C=c,
                solver="lbfgs",
                max_iter=2000,
            )

            model.fit(x_train, train_labels)
            dev_pred = model.predict(x_dev)
            dev_acc = accuracy_score(dev_labels, dev_pred)
            print(f" [dev] {language} |char_ngram | ngram_range=({min_n},{max_n}), C={c}: dev acc={dev_acc:.4f}")

            # Update best model if this one is better.
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                best_c = c
                best_ngram_range = (min_n, max_n)
                best_model = model
                selected_x_test = x_test

    if best_model is None or best_c is None or best_ngram_range is None or selected_x_test is None:
        raise ValueError("No model was trained. Check if ngram_ranges or c_values is empty.")

    # Evaluate best model on test set.
    test_pred = best_model.predict(selected_x_test)
    test_acc = accuracy_score(test_labels, test_pred)

    # Return experiment result.
    return ExperimentResult(
        language=language,
        model_name="LR",
        feature_name="char_ngram",
        dev_accuracy=float(best_dev_acc),
        test_accuracy=float(test_acc),
        hyperparameters={"C": best_c, "ngram_range": best_ngram_range},
    )


def run_part1_baselines(
    language_datasets: dict[str, object],
    c_values: list[float],
    ngram_ranges: list[tuple[int, int]],
    random_seed: int,
) -> list[ExperimentResult]:
    """Run both Part 1 baselines for all selected languages."""
    
    results: list[ExperimentResult] = []
    
    for language, dataset in language_datasets.items(): 
        print(f"Running baselines for {language}...")
        word_result = run_word_unigram(language, dataset, c_values, random_seed)
        print(f"  Best word unigram LR: dev acc={word_result.dev_accuracy:.4f}, test acc={word_result.test_accuracy:.4f}")
        results.append(word_result)
        
        char_result = run_char_ngram(language, dataset, ngram_ranges, c_values, random_seed)
        print(f"  Best char n-gram LR: dev acc={char_result.dev_accuracy:.4f}, test acc={char_result.test_accuracy:.4f}")
        results.append(char_result)
    return results
