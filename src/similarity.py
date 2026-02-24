"""Vectorized similarity scoring for batches of feature vectors."""

import numpy as np


# ---------------------------------------------------------------------------
# Vectorized implementations (no Python loops over N)
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute row-wise cosine similarity between two (N, D) arrays.

    Returns a length-N array of cosine similarities in [-1, 1].
    """
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)
    dot = np.sum(a * b, axis=1)
    # Avoid division by zero: if either norm is 0, similarity is 0
    denom = a_norm * b_norm
    denom = np.where(denom == 0, 1.0, denom)
    return dot / denom


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute row-wise Euclidean distance between two (N, D) arrays.

    Returns a length-N array of non-negative distances.
    """
    return np.linalg.norm(a - b, axis=1)


# ---------------------------------------------------------------------------
# Naive loop implementations (for benchmark baseline)
# ---------------------------------------------------------------------------

def cosine_similarity_loop(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity using a Python loop (slow baseline)."""
    n = a.shape[0]
    result = np.empty(n)
    for i in range(n):
        a_norm = np.linalg.norm(a[i])
        b_norm = np.linalg.norm(b[i])
        if a_norm == 0 or b_norm == 0:
            result[i] = 0.0
        else:
            result[i] = np.dot(a[i], b[i]) / (a_norm * b_norm)
    return result


def euclidean_distance_loop(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Euclidean distance using a Python loop (slow baseline)."""
    n = a.shape[0]
    result = np.empty(n)
    for i in range(n):
        result[i] = np.linalg.norm(a[i] - b[i])
    return result
