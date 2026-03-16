import re
from collections import Counter


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r"\b[a-z]+\b", text)
    return tokens


def build_vocab(tokens: list[str], min_count: int = 5, max_vocab_size=None):
    counter = Counter(tokens)

    vocab_items = [(word, freq) for word, freq in counter.items() if freq >= min_count]
    vocab_items.sort(key=lambda x: x[1], reverse=True)

    if max_vocab_size is not None:
        vocab_items = vocab_items[:max_vocab_size]

    word_to_idx = {word: idx for idx, (word, _) in enumerate(vocab_items)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    word_counts = {word: freq for word, freq in vocab_items}

    filtered_tokens = [word for word in tokens if word in word_to_idx]
    token_ids = [word_to_idx[word] for word in filtered_tokens]

    return word_to_idx, idx_to_word, word_counts, token_ids
