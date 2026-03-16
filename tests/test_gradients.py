import numpy as np
import pytest

from src.model import SkipGramNegativeSampling


EPS = 1e-6
ATOL = 1e-6
RTOL = 1e-4


def make_model():
    return SkipGramNegativeSampling(vocab_size=7, embedding_dim=5, seed=123)


def numerical_grad_scalar(loss_fn, param, index, eps=EPS):
    original = param[index]

    param[index] = original + eps
    loss_plus = loss_fn()

    param[index] = original - eps
    loss_minus = loss_fn()

    param[index] = original
    return (loss_plus - loss_minus) / (2.0 * eps)


def numerical_grad_row(loss_fn, matrix, row_idx, eps=EPS):
    grad = np.zeros_like(matrix[row_idx], dtype=np.float64)
    for col_idx in range(matrix.shape[1]):
        grad[col_idx] = numerical_grad_scalar(loss_fn, matrix, (row_idx, col_idx), eps=eps)
    return grad


def test_forward_backward_matches_finite_differences():
    model = make_model()

    center_idx = 1
    positive_idx = 2
    negative_indices = np.array([0, 3, 5], dtype=np.int64)

    loss, grad_center, grad_pos, grad_negs = model.forward_backward(
        center_idx=center_idx,
        positive_idx=positive_idx,
        negative_indices=negative_indices,
    )

    assert np.isfinite(loss)

    def loss_fn():
        return model.forward_backward(
            center_idx=center_idx,
            positive_idx=positive_idx,
            negative_indices=negative_indices,
        )[0]

    num_grad_center = numerical_grad_row(loss_fn, model.W_in, center_idx)
    num_grad_pos = numerical_grad_row(loss_fn, model.W_out, positive_idx)

    np.testing.assert_allclose(grad_center, num_grad_center, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(grad_pos, num_grad_pos, rtol=RTOL, atol=ATOL)

    for row_idx, neg_idx in enumerate(negative_indices):
        num_grad_neg = numerical_grad_row(loss_fn, model.W_out, int(neg_idx))
        np.testing.assert_allclose(grad_negs[row_idx], num_grad_neg, rtol=RTOL, atol=ATOL)


def test_duplicate_negative_indices_accumulate_correctly():
    model = make_model()

    center_idx = 1
    positive_idx = 2
    negative_indices = np.array([0, 0, 4], dtype=np.int64)

    _, grad_center, _, grad_negs = model.forward_backward(
        center_idx=center_idx,
        positive_idx=positive_idx,
        negative_indices=negative_indices,
    )

    def loss_fn():
        return model.forward_backward(
            center_idx=center_idx,
            positive_idx=positive_idx,
            negative_indices=negative_indices,
        )[0]

    num_grad_center = numerical_grad_row(loss_fn, model.W_in, center_idx)
    np.testing.assert_allclose(grad_center, num_grad_center, rtol=RTOL, atol=ATOL)

    grad_neg_0 = grad_negs[negative_indices == 0].sum(axis=0)
    num_grad_neg_0 = numerical_grad_row(loss_fn, model.W_out, 0)
    np.testing.assert_allclose(grad_neg_0, num_grad_neg_0, rtol=RTOL, atol=ATOL)

    grad_neg_4 = grad_negs[negative_indices == 4].sum(axis=0)
    num_grad_neg_4 = numerical_grad_row(loss_fn, model.W_out, 4)
    np.testing.assert_allclose(grad_neg_4, num_grad_neg_4, rtol=RTOL, atol=ATOL)


def test_small_update_reduces_loss_for_same_sample():
    model = make_model()

    center_idx = 1
    positive_idx = 2
    negative_indices = np.array([0, 3, 5], dtype=np.int64)

    loss_before, grad_center, grad_pos, grad_negs = model.forward_backward(
        center_idx=center_idx,
        positive_idx=positive_idx,
        negative_indices=negative_indices,
    )

    model.update(
        center_idx=center_idx,
        positive_idx=positive_idx,
        negative_indices=negative_indices,
        grad_center=grad_center,
        grad_pos=grad_pos,
        grad_negs=grad_negs,
        learning_rate=1e-3,
    )

    loss_after = model.forward_backward(
        center_idx=center_idx,
        positive_idx=positive_idx,
        negative_indices=negative_indices,
    )[0]

    assert loss_after < loss_before
