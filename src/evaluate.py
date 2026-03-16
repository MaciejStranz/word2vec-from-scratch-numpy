import numpy as np

from src.utils import cosine_similarity

import matplotlib.pyplot as plt


def plot_losses(losses: list[float], save_path: str | None = None):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training Loss")
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()
    
def get_nearest_neighbors(
    query_word: str,
    word_to_idx: dict,
    idx_to_word: dict,
    embeddings: np.ndarray,
    top_k: int = 5
):
    if query_word not in word_to_idx:
        return []

    query_idx = word_to_idx[query_word]
    query_vec = embeddings[query_idx]

    sims = cosine_similarity(query_vec, embeddings)
    sims[query_idx] = -np.inf

    nearest_indices = np.argsort(-sims)[:top_k]

    return [(idx_to_word[idx], float(sims[idx])) for idx in nearest_indices]


def print_neighbors(
    query_words: list[str],
    word_to_idx: dict,
    idx_to_word: dict,
    embeddings: np.ndarray,
    top_k: int = 5
):
    for word in query_words:
        neighbors = get_nearest_neighbors(
            query_word=word,
            word_to_idx=word_to_idx,
            idx_to_word=idx_to_word,
            embeddings=embeddings,
            top_k=top_k
        )

        print(f"\nNearest neighbors for '{word}':")
        if not neighbors:
            print("  [word not in vocabulary]")
            continue

        for neighbor_word, sim in neighbors:
            print(f"  {neighbor_word:15s} {sim:.4f}")
