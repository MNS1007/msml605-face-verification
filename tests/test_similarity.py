"""Unit tests for the similarity module."""

import numpy as np
import pytest

from src.similarity import (
    cosine_similarity,
    cosine_similarity_loop,
    euclidean_distance,
    euclidean_distance_loop,
)

TOLERANCE = 1e-6


class TestCosineSimilarity:
    def test_identical_vectors(self):
        a = np.array([[1.0, 2.0, 3.0]])
        result = cosine_similarity(a, a)
        np.testing.assert_allclose(result, [1.0], atol=TOLERANCE)

    def test_orthogonal_vectors(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[0.0, 1.0]])
        result = cosine_similarity(a, b)
        np.testing.assert_allclose(result, [0.0], atol=TOLERANCE)

    def test_opposite_vectors(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[-1.0, 0.0]])
        result = cosine_similarity(a, b)
        np.testing.assert_allclose(result, [-1.0], atol=TOLERANCE)

    def test_zero_vector(self):
        a = np.array([[0.0, 0.0]])
        b = np.array([[1.0, 2.0]])
        result = cosine_similarity(a, b)
        np.testing.assert_allclose(result, [0.0], atol=TOLERANCE)

    def test_batch(self):
        rng = np.random.default_rng(42)
        a = rng.standard_normal((100, 64))
        b = rng.standard_normal((100, 64))
        vec = cosine_similarity(a, b)
        loop = cosine_similarity_loop(a, b)
        np.testing.assert_allclose(vec, loop, atol=TOLERANCE)


class TestEuclideanDistance:
    def test_identical_vectors(self):
        a = np.array([[1.0, 2.0, 3.0]])
        result = euclidean_distance(a, a)
        np.testing.assert_allclose(result, [0.0], atol=TOLERANCE)

    def test_known_distance(self):
        a = np.array([[0.0, 0.0]])
        b = np.array([[3.0, 4.0]])
        result = euclidean_distance(a, b)
        np.testing.assert_allclose(result, [5.0], atol=TOLERANCE)

    def test_batch(self):
        rng = np.random.default_rng(42)
        a = rng.standard_normal((100, 64))
        b = rng.standard_normal((100, 64))
        vec = euclidean_distance(a, b)
        loop = euclidean_distance_loop(a, b)
        np.testing.assert_allclose(vec, loop, atol=TOLERANCE)
