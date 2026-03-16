import numpy as np

from src.model import SkipGramNegativeSampling
from src.train import train


def test_train_returns_one_loss_per_epoch_and_updates_weights():
    np.random.seed(123)

    vocab_size = 4
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0)] * 10
    probs = np.ones(vocab_size, dtype=np.float64) / vocab_size

    model = SkipGramNegativeSampling(vocab_size=vocab_size, embedding_dim=8, seed=123)
    w_in_before = model.W_in.copy()
    w_out_before = model.W_out.copy()

    losses = train(
        model=model,
        pairs=pairs,
        negative_sampling_probs=probs,
        vocab_size=vocab_size,
        num_negative_samples=2,
        learning_rate=0.05,
        epochs=3,
        verbose=False,
    )

    assert len(losses) == 3
    assert np.all(np.isfinite(losses))
    assert not np.allclose(model.W_in, w_in_before)
    assert not np.allclose(model.W_out, w_out_before)


def test_training_loss_decreases_on_tiny_repeated_corpus():
    np.random.seed(123)

    vocab_size = 4
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0)] * 50
    probs = np.ones(vocab_size, dtype=np.float64) / vocab_size

    model = SkipGramNegativeSampling(vocab_size=vocab_size, embedding_dim=8, seed=123)

    losses = train(
        model=model,
        pairs=pairs,
        negative_sampling_probs=probs,
        vocab_size=vocab_size,
        num_negative_samples=2,
        learning_rate=0.05,
        epochs=4,
        verbose=False,
    )

    assert losses[-1] < losses[0]
