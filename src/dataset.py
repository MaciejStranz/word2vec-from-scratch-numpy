import numpy as np


def generate_skipgram_pairs(token_ids: list[int], window_size: int) -> list[tuple[int, int]]:
    pairs = []

    for i, center_id in enumerate(token_ids):
        left = max(0, i - window_size)
        right = min(len(token_ids), i + window_size + 1)

        for j in range(left, right):
            if i == j:
                continue
            context_id = token_ids[j]
            pairs.append((center_id, context_id))

    return pairs


def build_negative_sampling_distribution(word_counts: dict, word_to_idx: dict, power: float = 0.75):
    vocab_size = len(word_to_idx)
    freqs = np.zeros(vocab_size, dtype=np.float64)

    for word, count in word_counts.items():
        idx = word_to_idx[word]
        freqs[idx] = count

    probs = freqs ** power
    probs /= probs.sum()

    return probs


def sample_negative_words(
    num_samples: int,
    vocab_size: int,
    probs: np.ndarray,
    positive_idx: int
) -> np.ndarray:
    negatives = np.random.choice(
        vocab_size,
        size=num_samples,
        replace=True,
        p=probs,
    )

    while True:
        mask = negatives == positive_idx
        if not np.any(mask):
            return negatives.astype(np.int64)

        negatives[mask] = np.random.choice(
            vocab_size,
            size=int(mask.sum()),
            replace=True,
            p=probs,
        )
