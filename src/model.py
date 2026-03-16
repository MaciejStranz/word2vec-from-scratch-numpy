import numpy as np

from src.utils import sigmoid


class SkipGramNegativeSampling:
    def __init__(self, vocab_size: int, embedding_dim: int, seed: int = 42):
        rng = np.random.default_rng(seed)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.W_in = rng.normal(0.0, 0.01, size=(vocab_size, embedding_dim))
        self.W_out = np.zeros((vocab_size, embedding_dim), dtype=np.float64)

    def forward_backward(
        self,
        center_idx: int,
        positive_idx: int,
        negative_indices: np.ndarray
    ):
        center_vec = self.W_in[center_idx]                # (D,)
        pos_vec = self.W_out[positive_idx]               # (D,)
        neg_vecs = self.W_out[negative_indices]          # (K, D)

        pos_score = pos_vec @ center_vec                 # scalar
        neg_scores = neg_vecs @ center_vec               # (K,)

        pos_sig = sigmoid(pos_score)                     # scalar
        neg_sigs = sigmoid(neg_scores)                   # (K,)

        loss = -np.log(pos_sig + 1e-10) - np.sum(np.log(1.0 - neg_sigs + 1e-10))

        grad_center = (pos_sig - 1.0) * pos_vec + np.sum(neg_sigs[:, None] * neg_vecs, axis=0)
        grad_pos = (pos_sig - 1.0) * center_vec
        grad_negs = neg_sigs[:, None] * center_vec[None, :]

        return loss, grad_center, grad_pos, grad_negs

    def update(
        self,
        center_idx: int,
        positive_idx: int,
        negative_indices: np.ndarray,
        grad_center: np.ndarray,
        grad_pos: np.ndarray,
        grad_negs: np.ndarray,
        learning_rate: float
    ):
        self.W_in[center_idx] -= learning_rate * grad_center
        self.W_out[positive_idx] -= learning_rate * grad_pos

        # Ważne: np.add.at poprawnie obsługuje powtarzające się indeksy
        np.add.at(self.W_out, negative_indices, -learning_rate * grad_negs)

    def get_input_embeddings(self) -> np.ndarray:
        return self.W_in

    def get_output_embeddings(self) -> np.ndarray:
        return self.W_out

    def get_final_embeddings(self) -> np.ndarray:
        return self.W_in
