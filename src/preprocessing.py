import re
from collections import Counter
from typing import Iterator


TOKEN_PATTERN = re.compile(r"[a-z]+")


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def iter_tokens(
    path: str,
    max_tokens: int | None = None,
    chunk_size: int = 4_000_000,
) -> Iterator[str]:
    emitted = 0
    carry = ""

    with open(path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            text = (carry + chunk).lower()

            cut = len(text)
            while cut > 0 and text[cut - 1].isalpha():
                cut -= 1

            carry = text[cut:]
            processable = text[:cut]

            for token in TOKEN_PATTERN.findall(processable):
                yield token
                emitted += 1
                if max_tokens is not None and emitted >= max_tokens:
                    return

        if carry:
            for token in TOKEN_PATTERN.findall(carry):
                yield token
                emitted += 1
                if max_tokens is not None and emitted >= max_tokens:
                    return


def load_tokens(path: str, max_tokens: int | None = None) -> list[str]:
    return list(iter_tokens(path, max_tokens=max_tokens))


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
