import numpy as np

from src.dataset import sample_negative_words


def train(
    model,
    pairs: list[tuple[int, int]],
    negative_sampling_probs: np.ndarray,
    vocab_size: int,
    num_negative_samples: int,
    learning_rate: float,
    epochs: int,
    verbose: bool = True
):
    losses = []

    for epoch in range(epochs):
        np.random.shuffle(pairs)
        epoch_loss = 0.0

        for center_idx, positive_idx in pairs:
            negative_indices = sample_negative_words(
                num_samples=num_negative_samples,
                vocab_size=vocab_size,
                probs=negative_sampling_probs,
                positive_idx=positive_idx
            )

            loss, grad_center, grad_pos, grad_negs = model.forward_backward(
                center_idx=center_idx,
                positive_idx=positive_idx,
                negative_indices=negative_indices
            )

            model.update(
                center_idx=center_idx,
                positive_idx=positive_idx,
                negative_indices=negative_indices,
                grad_center=grad_center,
                grad_pos=grad_pos,
                grad_negs=grad_negs,
                learning_rate=learning_rate
            )

            epoch_loss += loss

        avg_loss = epoch_loss / len(pairs)
        losses.append(avg_loss)

        if verbose:
            print(f"Epoch {epoch + 1}/{epochs} - avg loss: {avg_loss:.4f}")

    return losses
