import numpy as np


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    flat = x.reshape(-1)
    out = np.empty_like(flat, dtype=np.float64)

    pos_mask = flat >= 0.0
    out[pos_mask] = 1.0 / (1.0 + np.exp(-flat[pos_mask]))

    exp_x = np.exp(flat[~pos_mask])
    out[~pos_mask] = exp_x / (1.0 + exp_x)

    out = out.reshape(x.shape)
    return float(out) if out.ndim == 0 else out


def softplus(x):
    return np.logaddexp(0.0, x)


def cosine_similarity(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    vec_norm = np.linalg.norm(vec) + 1e-12
    matrix_norm = np.linalg.norm(matrix, axis=1) + 1e-12
    return (matrix @ vec) / (matrix_norm * vec_norm)

