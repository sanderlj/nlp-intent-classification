"""Standalone multinomial Naive Bayes classifier."""

from __future__ import annotations

from collections import Counter, defaultdict
from math import log


class MultinomialNaiveBayes:
    """Naive Bayes classifier over token-count features."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.class_log_priors: dict[str, float] = {}
        self.token_log_probs: dict[str, dict[str, float]] = {}
        self.vocabulary: set[str] = set()

    def fit(self, documents: list[list[str]], labels: list[str]) -> None:
        """Estimate class priors and token likelihoods."""
        
        if len(documents) != len(labels):
            raise ValueError("Number of documents and labels must match.")
        if not documents:
            raise ValueError("Training data cannot be empty.")
        if self.alpha <= 0:
            raise ValueError("Smoothing parameter alpha must be positive.")
        
        # Reset state to prevent repeated calls to fit from accumulating state.
        self.class_log_priors = {}
        self.token_log_probs = {}   
        self.vocabulary = set()
        
        # Global vocabulary is the set of all unique tokens across all documents.
        for document in documents:
            self.vocabulary.update(document)
        
        # Need at least one token in the vocabulary to compute probabilities.
        vocab_size = len(self.vocabulary)
        if vocab_size == 0:
            raise ValueError("Vocabulary cannot be empty.")

        # Compute class priors P(class) 
        label_counts = Counter(labels)
        num_documents = len(labels)
        for label, count in label_counts.items():
            self.class_log_priors[label] = log(count / num_documents)
        
        # Count tokens frequencies per class and compute log probabilities with Laplace smoothing.
        token_counts_by_class = count_tokens_by_class(documents, labels)
        
        for label, token_counts in token_counts_by_class.items():
            total_tokens_in_class = sum(token_counts.values())
            denom = total_tokens_in_class + self.alpha * vocab_size
            
            class_token_log_probs = {}
            for token in self.vocabulary:
                count = token_counts.get(token, 0)
                class_token_log_probs[token] = log((count + self.alpha) / denom)
            
            class_token_log_probs["<UNK>"] = log(self.alpha / denom)
            self.token_log_probs[label] = class_token_log_probs


    def predict_one(self, document: list[str]) -> str:
        """Predict the label for one tokenized document."""
        
        if not self.class_log_priors or not self.token_log_probs:
            raise ValueError("Model has not been fitted yet.")
        
        best_label = None
        best_score = float("-inf")
        
        for label, log_prior in self.class_log_priors.items():
            score = log_prior
            class_token_log_probs = self.token_log_probs[label]
            if "<UNK>" not in class_token_log_probs:
                raise ValueError(f"Missing <UNK> probability for class '{label}'.")
            unk_log_prob = class_token_log_probs["<UNK>"]
            
            for token in document:
                token_log_prob = class_token_log_probs[token] if token in self.vocabulary else unk_log_prob
                score += token_log_prob
            
            if score > best_score:
                best_score = score
                best_label = label
                
        if best_label is None:
            raise ValueError("No valid label found for prediction.")
        
        return best_label
        
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
