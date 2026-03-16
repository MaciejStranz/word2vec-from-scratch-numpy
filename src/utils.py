import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -10, 10)
    return 1.0 / (1.0 + np.exp(-x))


def cosine_similarity(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    vec_norm = np.linalg.norm(vec) + 1e-12
    matrix_norm = np.linalg.norm(matrix, axis=1) + 1e-12
    return (matrix @ vec) / (matrix_norm * vec_norm)
