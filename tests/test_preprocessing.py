from src.preprocessing import build_vocab, iter_tokens, load_tokens


def test_iter_tokens_handles_chunk_boundaries_and_lowercases(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("Hello WORLD alphaBeta gamma.\nDelta", encoding="utf-8")

    tokens = list(iter_tokens(str(path), chunk_size=5))

    assert tokens == ["hello", "world", "alphabeta", "gamma", "delta"]


def test_load_tokens_respects_max_tokens(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("one TWO three four five six", encoding="utf-8")

    tokens = load_tokens(str(path), max_tokens=4)

    assert tokens == ["one", "two", "three", "four"]


def test_build_vocab_filters_by_min_count_and_returns_token_ids():
    tokens = ["cat", "dog", "cat", "bird", "dog", "cat"]

    word_to_idx, idx_to_word, word_counts, token_ids = build_vocab(tokens, min_count=2)

    assert set(word_to_idx.keys()) == {"cat", "dog"}
    assert idx_to_word[word_to_idx["cat"]] == "cat"
    assert idx_to_word[word_to_idx["dog"]] == "dog"
    assert word_counts == {"cat": 3, "dog": 2}
    assert token_ids == [
        word_to_idx["cat"],
        word_to_idx["dog"],
        word_to_idx["cat"],
        word_to_idx["dog"],
        word_to_idx["cat"],
    ]
