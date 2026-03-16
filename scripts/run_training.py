import numpy as np

from src.config import (
    EMBEDDING_DIM,
    WINDOW_SIZE,
    NUM_NEGATIVE_SAMPLES,
    LEARNING_RATE,
    EPOCHS,
    MIN_COUNT,
    MAX_VOCAB_SIZE,
    MAX_TOKENS,
    SEED,
)
from src.preprocessing import load_text, tokenize, build_vocab
from src.dataset import generate_skipgram_pairs, build_negative_sampling_distribution
from src.model import SkipGramNegativeSampling
from src.train import train
from src.evaluate import print_neighbors, plot_losses


def main():
    np.random.seed(SEED)

    text = load_text("data/text8") 
    tokens = tokenize(text)

    if MAX_TOKENS is not None:
        tokens = tokens[:MAX_TOKENS]

    word_to_idx, idx_to_word, word_counts, token_ids = build_vocab(
        tokens=tokens,
        min_count=MIN_COUNT,
        max_vocab_size=MAX_VOCAB_SIZE,
    )

    vocab_size = len(word_to_idx)

    print(f"Number of tokens after filtering: {len(token_ids)}")
    print(f"Vocabulary size: {vocab_size}")

    pairs = generate_skipgram_pairs(token_ids, window_size=WINDOW_SIZE)
    print(f"Number of training pairs: {len(pairs)}")

    neg_probs = build_negative_sampling_distribution(
        word_counts=word_counts,
        word_to_idx=word_to_idx,
        power=0.75,
    )

    model = SkipGramNegativeSampling(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        seed=SEED,
    )

    losses = train(
        model=model,
        pairs=pairs,
        negative_sampling_probs=neg_probs,
        vocab_size=vocab_size,
        num_negative_samples=NUM_NEGATIVE_SAMPLES,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
    )

    print("\nTraining finished.")
    print("Loss history:", losses)

    embeddings = model.get_final_embeddings()

    sample_words = ["king", "queen", "man", "woman", "city", "love"]
    print_neighbors(
        query_words=sample_words,
        word_to_idx=word_to_idx,
        idx_to_word=idx_to_word,
        embeddings=embeddings,
        top_k=5,
    )

    plot_losses(losses, save_path="training_loss.png")



if __name__ == "__main__":
    main()
