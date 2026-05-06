"""Experiment helpers for Assignment 1. 
This module contains code to run the experiments for Part 1 of the assignment.
"""


from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.data_utils import LanguageDataset
from src.evaluation import top_confused_pairs
from bpe import BPETokenizer
from nb import MultinomialNaiveBayes

from src.features import (
    concatenate_feature_blocks,
    fit_bpe_bigram_vocabulary,
    fit_char_ngram_vocabulary,
    fit_word_unigram_vocabulary,
    transform_bpe_bigram_counts,
    transform_length_structure_features,
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
            random_state=random_seed,
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

        for c in c_values:
            model = LogisticRegression(
                C=c,
                solver="lbfgs",
                max_iter=2000,
                random_state=random_seed,
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
                selected_x_test = transform_char_ngram_counts(test_texts, vocab, min_n, max_n)

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


def run_part3_nb(
    language_datasets: dict[str, LanguageDataset],
    alpha_values: list[float],
    k_values: list[int],
    random_seed: int,
) -> list[ExperimentResult]:
    """Run the Part 3 NB experiment for all selected languages."""
    
    results: list[ExperimentResult] = []
    
    for language, dataset in language_datasets.items():
        print(f"Running Naive Bayes for {language}...")
        
        train_labels = dataset.train.labels
        dev_labels = dataset.dev.labels
        test_labels = dataset.test.labels
        train_texts = dataset.train.texts
        dev_texts = dataset.dev.texts
        test_texts = dataset.test.texts
        
        word_list = []
        for text in train_texts:
            word_list.extend(text.split())
        
        best_dev_acc = -1.0
        best_k = None
        best_alpha = None
        
        for k in k_values:
            tokenizer = BPETokenizer()
            print(f"  [NB] Tuning for k={k}...")
            tokenizer.train(word_list, k)
            train_docs = [tokenizer.encode(text) for text in train_texts]
            dev_docs = [tokenizer.encode(text) for text in dev_texts]
            
            for alpha in alpha_values:
                print(f"    [NB] Tuning for alpha={alpha}...")
                model = MultinomialNaiveBayes(alpha=alpha)
                model.fit(train_docs, train_labels)
                dev_pred = model.predict(dev_docs)
                dev_acc = accuracy_score(dev_labels, dev_pred)  
                print(f"      [NB] k={k}, alpha={alpha}: dev acc={dev_acc:.4f}")
                
                if dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                    best_k = k
                    best_alpha = alpha
        if best_k is None or best_alpha is None:
            raise ValueError("No valid hyperparameters found. Check if k_values or alpha_values is empty.")
        print(f"  Best NB hyperparameters for {language}: k={best_k}, alpha={best_alpha}, dev acc={best_dev_acc:.4f}")
        
        best_tokenizer = BPETokenizer()
        best_tokenizer.train(word_list, best_k)
        best_train_docs = [best_tokenizer.encode(text) for text in train_texts]
        best_test_docs = [best_tokenizer.encode(text) for text in test_texts]
        best_model = MultinomialNaiveBayes(alpha=best_alpha)
        best_model.fit(best_train_docs, train_labels)
        test_pred = best_model.predict(best_test_docs)
        test_acc = accuracy_score(test_labels, test_pred)
        print(f"  Test accuracy for {language}: {test_acc:.4f}")
        
        # Return experiment result.
        results.append(ExperimentResult(
            language=language,
            model_name="NB",
            feature_name="BPE",
            dev_accuracy=float(best_dev_acc),
            test_accuracy=float(test_acc),
            hyperparameters={"k": best_k, "alpha": best_alpha},
        ))

    return results


def build_feature_engineering_matrices(train_texts: list[str], other_texts: list[str], train_words: list[str], k: int, feature_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Helper function to build feature matrices for Part 4 feature engineering."""
    
    tokenizer = BPETokenizer()
    tokenizer.train(train_words, k)
    train_docs = [tokenizer.encode(text) for text in train_texts]
    other_docs = [tokenizer.encode(text) for text in other_texts]
    
    bpe_vocab: dict[str, int] = {}
    for doc in train_docs:
        for token in doc:
            if token not in bpe_vocab:
                bpe_vocab[token] = len(bpe_vocab)
    
    x_train_bpe = np.zeros((len(train_docs), len(bpe_vocab)), dtype=np.int32)
    x_other_bpe = np.zeros((len(other_docs), len(bpe_vocab)), dtype=np.int32)
    
    for i, doc in enumerate(train_docs):
        for tok in doc:
            if tok in bpe_vocab:
                x_train_bpe[i, bpe_vocab[tok]] += 1
    for i, doc in enumerate(other_docs):
        for tok in doc:
            if tok in bpe_vocab:
                x_other_bpe[i, bpe_vocab[tok]] += 1
    
    train_blocks = [x_train_bpe]
    other_blocks = [x_other_bpe]
                    
    if feature_name in {"bpe_counts_plus_bigrams", "bpe_counts_plus_bigrams_plus_length"}:
        bigram_vocab = fit_bpe_bigram_vocabulary(train_docs)
        train_bigram_features = transform_bpe_bigram_counts(train_docs, bigram_vocab)
        other_bigram_features = transform_bpe_bigram_counts(other_docs, bigram_vocab)
        train_blocks.append(train_bigram_features)
        other_blocks.append(other_bigram_features)

    if feature_name in {"bpe_counts_plus_length", "bpe_counts_plus_bigrams_plus_length"}:
        train_length_features = transform_length_structure_features(train_texts)
        other_length_features = transform_length_structure_features(other_texts)
        train_blocks.append(train_length_features)
        other_blocks.append(other_length_features)

    x_train = concatenate_feature_blocks(train_blocks)
    x_other = concatenate_feature_blocks(other_blocks)
        
    return x_train, x_other

def run_part4_feature_engineering(
    language_datasets: dict[str, LanguageDataset],
    k_values: list[int],
    c_values: list[float],
    random_seed: int,
) -> list[ExperimentResult]:
    """Run Part 4 LR + BPE feature engineering with ablations.

    Ablation plan:
    1) BPE counts (base)
    2) BPE + BPE bigrams
    3) BPE + length_structure_features
    4) BPE + both features
    """
    results: list[ExperimentResult] = []
    feature_sets = [
        "bpe_counts",
        "bpe_counts_plus_bigrams",
        "bpe_counts_plus_length",
        "bpe_counts_plus_bigrams_plus_length",
    ]

    for language, dataset in language_datasets.items():
        print(f"Running Part 4 feature engineering for {language}...")
        train_texts = dataset.train.texts
        dev_texts = dataset.dev.texts
        test_texts = dataset.test.texts
        train_labels = dataset.train.labels
        dev_labels = dataset.dev.labels
        test_labels = dataset.test.labels

        train_words: list[str] = []
        for text in train_texts:
            train_words.extend(text.split())

        for feature_name in feature_sets:
            print(f"  [Part4] feature_set={feature_name}")
            best_dev_acc = -1.0
            best_k = None
            best_c = None

            for k in k_values:
                x_train, x_dev = build_feature_engineering_matrices(train_texts, dev_texts, train_words, k, feature_name)
                for c_value in c_values:
                    print(f"    Tuning for k={k}, C={c_value}...")
                    model = LogisticRegression(
                        C=c_value,
                        solver="lbfgs",
                        max_iter=5000,
                        random_state=random_seed,
                    )
                    model.fit(x_train, train_labels)
                    dev_pred = model.predict(x_dev)
                    dev_acc = accuracy_score(dev_labels, dev_pred)
                    print(f"      dev acc={dev_acc:.4f}")
                    if dev_acc > best_dev_acc:
                        best_dev_acc = dev_acc
                        best_k = k
                        best_c = c_value
            print(f"  Best hyperparameters for {language} with feature set {feature_name}: k={best_k}, C={best_c}, dev acc={best_dev_acc:.4f}")
            if best_k is None or best_c is None:
                raise ValueError("No valid hyperparameters found. Check if k_values or c_values is empty.")
            
            x_train_best, x_test_best = build_feature_engineering_matrices(
                train_texts,
                test_texts,
                train_words,
                best_k,
                feature_name
            )
            
            best_model = LogisticRegression(
                C=best_c,
                solver="lbfgs",
                max_iter=5000,
                random_state=random_seed
            )
            
            best_model.fit(x_train_best, train_labels)
            test_pred = best_model.predict(x_test_best)
            test_acc = accuracy_score(test_labels, test_pred)
            print(
                f"  [Part4][{language}][{feature_name}] "
                f"best_k={best_k}, best_C={best_c}, "
                f"dev={best_dev_acc:.4f}, test={test_acc:.4f}"
            )

            results.append(ExperimentResult(
                language=language,
                model_name="LR",
                feature_name=feature_name,
                dev_accuracy=float(best_dev_acc),
                test_accuracy=float(test_acc),
                hyperparameters={"k": best_k, "C": best_c},
            ))

    return results

def build_part4_feature_names(
    train_texts: list[str],
    train_words: list[str],
    k: int,
    feature_name: str,
) -> list[str]:
    tokenizer = BPETokenizer()
    tokenizer.train(train_words, k)
    train_docs = [tokenizer.encode(text) for text in train_texts]

    # Base BPE token names
    bpe_vocab: dict[str, int] = {}
    for doc in train_docs:
        for tok in doc:
            if tok not in bpe_vocab:
                bpe_vocab[tok] = len(bpe_vocab)
    bpe_names = [""] * len(bpe_vocab)
    for tok, idx in bpe_vocab.items():
        bpe_names[idx] = f"bpe={tok}"

    names = list(bpe_names)

    # BPE bigram names
    if feature_name in {"bpe_counts_plus_bigrams", "bpe_counts_plus_bigrams_plus_length"}:
        bigram_vocab = fit_bpe_bigram_vocabulary(train_docs)
        bigram_names = [""] * len(bigram_vocab)
        for bg, idx in bigram_vocab.items():
            bigram_names[idx] = f"bpe_bigram={bg}"
        names.extend(bigram_names)

    # Length names
    if feature_name in {"bpe_counts_plus_length", "bpe_counts_plus_bigrams_plus_length"}:
        names.extend(["len_num_tokens", "len_num_chars", "len_avg_token_len"])

    return names

def collect_part3_alpha_curve(
    language_datasets: dict[str, LanguageDataset],
    fixed_k_by_language: dict[str, int],
    alpha_values: list[float],
) -> list[dict]:
    """Run a targeted dev-only NB pass for alpha-curve plotting.

    Uses fixed k per language and evaluates dev accuracy for each alpha.
    No test evaluation is performed in this helper.
    """
    curve_rows: list[dict] = []

    for language, dataset in language_datasets.items():
        if language not in fixed_k_by_language:
            raise ValueError(f"Missing fixed k for language '{language}'.")

        k_value = fixed_k_by_language[language]
        train_texts = dataset.train.texts
        train_labels = dataset.train.labels
        dev_texts = dataset.dev.texts
        dev_labels = dataset.dev.labels

        train_words: list[str] = []
        for text in train_texts:
            train_words.extend(text.split())

        tokenizer = BPETokenizer()
        tokenizer.train(train_words, k_value)
        train_docs = [tokenizer.encode(text) for text in train_texts]
        dev_docs = [tokenizer.encode(text) for text in dev_texts]

        for alpha in alpha_values:
            model = MultinomialNaiveBayes(alpha=alpha)
            model.fit(train_docs, train_labels)
            dev_pred = model.predict(dev_docs)
            dev_acc = float(accuracy_score(dev_labels, dev_pred))

            curve_rows.append(
                {
                    "language": language,
                    "k": k_value,
                    "alpha": alpha,
                    "dev_accuracy": dev_acc,
                }
            )

    return curve_rows

def run_error_analysis_for_model(
    language: str,
    dataset: LanguageDataset,
    model_kind: str,
    config: dict,
    random_seed: int,
    top_k_pairs: int = 3,
    top_n_features: int = 10,
) -> dict:
    
    """Run error analysis for a given model configuration."""
    train_texts = dataset.train.texts
    train_labels = dataset.train.labels
    test_texts = dataset.test.texts
    test_labels = dataset.test.labels
    
    if model_kind == "part1_char_lr":
        ngram_range = config["ngram_range"]
        c_value = config["C"]
        vocab = fit_char_ngram_vocabulary(train_texts, ngram_range[0], ngram_range[1])
        x_train = transform_char_ngram_counts(train_texts, vocab, ngram_range[0], ngram_range[1])
        x_test = transform_char_ngram_counts(test_texts, vocab, ngram_range[0], ngram_range[1])
        
        model = LogisticRegression(
            C=c_value,
            solver="lbfgs",
            max_iter=5000,
            random_state=random_seed,
        )
        
        model.fit(x_train, train_labels)
        y_pred = model.predict(x_test).tolist()
        test_acc = float(accuracy_score(test_labels, y_pred))
        
        feature_names = [""] * len(vocab)
        for ngram, idx in vocab.items():
            feature_names[idx] = f"char_ngram={ngram}"
        
    elif model_kind == "part4_bpe_lr":
        k_value = config["k"]
        c_value = config["C"]
        feature_name = config["feature_name"]
        train_words = []
        for text in train_texts:
            train_words.extend(text.split())
        x_train, x_test = build_feature_engineering_matrices(
            train_texts=train_texts,
            other_texts=test_texts,
            train_words=train_words,
            k=k_value,
            feature_name=feature_name,
        )
        model = LogisticRegression(
            C=c_value,
            solver="lbfgs",
            max_iter=5000,
            random_state=random_seed,
        )
        model.fit(x_train, train_labels)
        y_pred = model.predict(x_test).tolist()
        test_acc = float(accuracy_score(test_labels, y_pred))
        feature_names = build_part4_feature_names(
            train_texts=train_texts,
            train_words=train_words,
            k=k_value,
            feature_name=feature_name,
        )
    else:
        raise ValueError(f"Unsupported model kind: {model_kind}")
    
    pairs = top_confused_pairs(test_labels, y_pred, top_k=top_k_pairs)
    
    feature_evidence: dict[str, dict] = {}
    class_to_idx = {label: i for i, label in enumerate(model.classes_)}
    
    for (a, b), _count in pairs:
        key = f"{a}__{b}"
        idx_a = class_to_idx[a]
        idx_b = class_to_idx[b]
        
        wa = model.coef_[idx_a]
        wb = model.coef_[idx_b]
        
        top_a_idx = np.argsort(wa)[-top_n_features:][::-1]
        top_b_idx = np.argsort(wb)[-top_n_features:][::-1]
        
        top_a = [{"feature": feature_names[i], "weight": float(wa[i])} for i in top_a_idx]
        top_b = [{"feature": feature_names[i], "weight": float(wb[i])} for i in top_b_idx]

    
        set_a = {x["feature"] for x in top_a}
        set_b = {x["feature"] for x in top_b}
        overlap = sorted(set_a & set_b)

        feature_evidence[key] = {
            a: top_a,
            b: top_b,
            "overlap_top_features": overlap,
        }
    
    examples_by_pair: dict[str, list[dict[str, str]]] = {}
    for (a, b), _count in pairs:
        key = f"{a}__{b}"
        examples: list[dict[str, str]] = []

        for i, (yt, yp) in enumerate(zip(test_labels, y_pred)):
            if (yt == a and yp == b) or (yt == b and yp == a):
                examples.append(
                    {
                        "true": yt,
                        "pred": yp,
                        "text": test_texts[i],
                    }
                )
            if len(examples) >= 3:
                break

        examples_by_pair[key] = examples

    return {
        "language": language,
        "model_kind": model_kind,
        "config": config,
        "test_accuracy": test_acc,
        "top_confused_pairs": pairs,
        "examples": examples_by_pair,
        "feature_evidence": feature_evidence,
    }
    
    

