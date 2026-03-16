import argparse
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
from src.preprocessing import load_tokens, build_vocab
from src.dataset import generate_skipgram_pairs, build_negative_sampling_distribution
from src.model import SkipGramNegativeSampling
from src.train import train
from src.evaluate import print_neighbors, plot_losses


def parse_args():
    parser = argparse.ArgumentParser(description="Train Word2Vec (Skip-Gram + Negative Sampling) in NumPy.")
    parser.add_argument("--data-path", default="data/text8")
    parser.add_argument("--embedding-dim", type=int, default=EMBEDDING_DIM)
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--num-negative-samples", type=int, default=NUM_NEGATIVE_SAMPLES)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--min-count", type=int, default=MIN_COUNT)
    parser.add_argument("--max-vocab-size", type=int, default=MAX_VOCAB_SIZE)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    tokens = load_tokens(args.data_path, max_tokens=args.max_tokens)

    word_to_idx, idx_to_word, word_counts, token_ids = build_vocab(
        tokens=tokens,
        min_count=args.min_count,
        max_vocab_size=args.max_vocab_size,
    )

    vocab_size = len(word_to_idx)

    print(f"Number of tokens after filtering: {len(token_ids)}")
    print(f"Vocabulary size: {vocab_size}")

    pairs = generate_skipgram_pairs(token_ids, window_size=args.window_size)
    print(f"Number of training pairs: {len(pairs)}")

    neg_probs = build_negative_sampling_distribution(
        word_counts=word_counts,
        word_to_idx=word_to_idx,
        power=0.75,
    )

    model = SkipGramNegativeSampling(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        seed=args.seed,
    )

    losses = train(
        model=model,
        pairs=pairs,
        negative_sampling_probs=neg_probs,
        vocab_size=vocab_size,
        num_negative_samples=args.num_negative_samples,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
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

    if not args.no_plot:
        plot_losses(losses, save_path="training_loss.png")


if __name__ == "__main__":
    main()
