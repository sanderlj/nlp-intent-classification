"""Standalone Byte-Pair Encoding tokenizer for Assignment 1."""

from __future__ import annotations

from collections import Counter

class BPETokenizer:
    """Simple BPE tokenizer with train and encode methods."""

    def __init__(self, end_of_word: str = "_") -> None:
        self.end_of_word = end_of_word
        self.merges: list[tuple[str, str]] = []
        self.vocab: Counter[tuple[str, ...]] = Counter()

    def train(self, words: list[str], num_merges: int) -> None:
        """Learn BPE merges from a training corpus."""
        
        self.merges = []
        self.vocab = Counter()
        
        for word in words:
            word = word.strip()
            if not word:
                continue
            token_tuple = tuple(word) + (self.end_of_word,)
            self.vocab[token_tuple] += 1
            
        for step in range(num_merges):
            pair_freqs = get_pair_frequencies(self.vocab)
            if not pair_freqs:
                break
            most_common_pair, _ = pair_freqs.most_common(1)[0]
            
            self.merges.append(most_common_pair)
            self.vocab = merge_pair_in_vocab(self.vocab, most_common_pair)

    def encode_word(self, word: str) -> list[str]:
        """Encode a single word using learned merges."""
        
        word = word.strip()
        if not word:
            return []
        
        symbols: list[str] = list(word) + [self.end_of_word]
        
        for a, b in self.merges:
            merged = a + b
            new_symbols: list[str] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        return symbols
        

    def encode(self, text: str) -> list[str]:
        """Encode an utterance by applying BPE word by word."""
        
        tokens: list[str] = []
        for word in text.strip().split():
            tokens.extend(self.encode_word(word))
        return tokens


def get_pair_frequencies(vocab: Counter[tuple[str, ...]]) -> Counter[tuple[str, str]]:
    """Count adjacent symbol-pair frequencies in a BPE vocabulary."""
    
    pair_counts: Counter[tuple[str, str]] = Counter()
    for symbols, freq in vocab.items():
        for index in range(len(symbols) - 1):
            pair_counts[(symbols[index], symbols[index + 1])] += freq
    return pair_counts


def merge_pair_in_vocab(
    vocab: Counter[tuple[str, ...]],
    pair: tuple[str, str],
) -> Counter[tuple[str, ...]]:
    """Merge one symbol pair across all words in the vocabulary."""
    a, b = pair
    merged = a + b
    new_vocab: Counter[tuple[str, ...]] = Counter()
    
    for symbols, freq in vocab.items():
        new_symbols: list[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new_symbols.append(merged)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_vocab[tuple(new_symbols)] += freq
    
    return new_vocab
