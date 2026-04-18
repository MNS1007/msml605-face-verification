"""Unit tests for src/confidence.py — calibrated confidence."""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.confidence import calibrated_confidence


# ------------------------------------------------------------------
# Scalar tests
# ------------------------------------------------------------------

class TestScalarConfidence:
    """Tests on single float scores."""

    def test_at_threshold_returns_half(self):
        """Score exactly at threshold -> confidence = 0.5 (borderline)."""
        assert calibrated_confidence(0.35, threshold=0.35) == pytest.approx(0.5)

    def test_max_score_returns_one(self):
        """Score at max (1.0) -> confidence = 1.0."""
        assert calibrated_confidence(1.0, threshold=0.35) == pytest.approx(1.0)

    def test_min_score_returns_one(self):
        """Score at min (-1.0) -> confidence = 1.0."""
        assert calibrated_confidence(-1.0, threshold=0.35) == pytest.approx(1.0)

    def test_above_threshold_higher_than_half(self):
        """Score above threshold -> confidence > 0.5."""
        c = calibrated_confidence(0.6, threshold=0.35)
        assert c > 0.5

    def test_below_threshold_higher_than_half(self):
        """Score below threshold -> confidence > 0.5."""
        c = calibrated_confidence(0.1, threshold=0.35)
        assert c > 0.5

    def test_symmetry_near_boundary(self):
        """Equal distance above/below threshold gives same relative offset."""
        t = 0.0  # symmetric threshold for cosine [-1, 1]
        c_above = calibrated_confidence(0.3, threshold=t)
        c_below = calibrated_confidence(-0.3, threshold=t)
        assert c_above == pytest.approx(c_below)

    def test_confidence_increases_with_distance(self):
        """Farther from threshold -> higher confidence."""
        t = 0.35
        c_near = calibrated_confidence(0.40, threshold=t)
        c_far = calibrated_confidence(0.80, threshold=t)
        assert c_far > c_near

    def test_returns_float_for_scalar(self):
        result = calibrated_confidence(0.5, threshold=0.3)
        assert isinstance(result, float)


# ------------------------------------------------------------------
# Array tests
# ------------------------------------------------------------------

class TestArrayConfidence:
    """Tests on numpy arrays of scores."""

    def test_array_shape_preserved(self):
        scores = np.array([0.1, 0.35, 0.6, 0.9])
        conf = calibrated_confidence(scores, threshold=0.35)
        assert conf.shape == (4,)

    def test_all_values_in_range(self):
        scores = np.linspace(-1.0, 1.0, 100)
        conf = calibrated_confidence(scores, threshold=0.35)
        assert np.all(conf >= 0.5)
        assert np.all(conf <= 1.0)

    def test_batch_matches_scalar(self):
        scores = np.array([-0.5, 0.0, 0.35, 0.7, 1.0])
        t = 0.35
        batch = calibrated_confidence(scores, threshold=t)
        for i, s in enumerate(scores):
            assert batch[i] == pytest.approx(calibrated_confidence(s, threshold=t))


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------

class TestEdgeCases:
    def test_threshold_at_max(self):
        """Threshold = 1.0 (everything predicted different)."""
        c = calibrated_confidence(0.5, threshold=1.0)
        assert 0.5 <= c <= 1.0

    def test_threshold_at_min(self):
        """Threshold = -1.0 (everything predicted same)."""
        c = calibrated_confidence(0.5, threshold=-1.0)
        assert 0.5 <= c <= 1.0

    def test_custom_score_range(self):
        """Non-default score range [0, 1]."""
        c = calibrated_confidence(0.8, threshold=0.5, score_min=0.0, score_max=1.0)
        assert 0.5 < c <= 1.0
