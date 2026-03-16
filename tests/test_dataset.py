import numpy as np

from src.dataset import (
    build_negative_sampling_distribution,
    generate_skipgram_pairs,
    sample_negative_words,
)


def test_generate_skipgram_pairs_window_1():
    token_ids = [0, 1, 2, 3]

    pairs = generate_skipgram_pairs(token_ids, window_size=1)

    expected = [
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
        (2, 3),
        (3, 2),
    ]
    assert pairs == expected


def test_build_negative_sampling_distribution_is_normalized_and_frequency_ordered():
    word_counts = {"a": 10, "b": 5, "c": 1}
    word_to_idx = {"a": 0, "b": 1, "c": 2}

    probs = build_negative_sampling_distribution(word_counts, word_to_idx, power=0.75)

    np.testing.assert_allclose(probs.sum(), 1.0)
    assert probs[0] > probs[1] > probs[2]


def test_sample_negative_words_never_returns_positive_idx():
    np.random.seed(123)

    probs = np.array([0.7, 0.1, 0.1, 0.1], dtype=np.float64)
    samples = sample_negative_words(
        num_samples=100,
        vocab_size=4,
        probs=probs,
        positive_idx=0,
    )

    assert samples.dtype == np.int64
    assert len(samples) == 100
    assert np.all(samples != 0)
    assert np.all((0 <= samples) & (samples < 4))
