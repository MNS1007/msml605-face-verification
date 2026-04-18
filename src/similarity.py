"""Vectorized similarity scoring for batches of feature vectors (embeddings)."""

import numpy as np


# ---------------------------------------------------------------------------
# Vectorized implementations (no Python loops over N)
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute row-wise cosine similarity between two (N, D) embedding arrays.

    Returns a length-N array of cosine similarities in [-1, 1].
    """
    # Compute norms
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)

    # Dot product
    dot = np.sum(a * b, axis=1)

    # Avoid division by zero
    denom = a_norm * b_norm
    denom = np.where(denom == 0, 1.0, denom)

    sim = dot / denom

    # Clip for numerical stability (important for embeddings)
    return np.clip(sim, -1.0, 1.0)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute row-wise Euclidean distance between two (N, D) embedding arrays.

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
            val = np.dot(a[i], b[i]) / (a_norm * b_norm)
            result[i] = np.clip(val, -1.0, 1.0)
    return result


def euclidean_distance_loop(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Euclidean distance using a Python loop (slow baseline)."""
    n = a.shape[0]
    result = np.empty(n)
    for i in range(n):
        result[i] = np.linalg.norm(a[i] - b[i])
    return result