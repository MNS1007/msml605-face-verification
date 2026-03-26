"""Unit tests for the evaluation module."""

import numpy as np
import pytest

from src.evaluation import (
    apply_threshold,
    compute_metrics,
    confusion_matrix,
    roc_curve_data,
    select_threshold,
    threshold_sweep,
)


class TestApplyThreshold:
    def test_higher_is_same(self):
        scores = np.array([0.2, 0.5, 0.8, 0.9])
        preds = apply_threshold(scores, threshold=0.5, higher_is_same=True)
        np.testing.assert_array_equal(preds, [0, 1, 1, 1])

    def test_lower_is_same(self):
        scores = np.array([0.1, 0.3, 0.6, 0.9])
        preds = apply_threshold(scores, threshold=0.5, higher_is_same=False)
        np.testing.assert_array_equal(preds, [1, 1, 0, 0])

    def test_exact_threshold_included(self):
        scores = np.array([0.5])
        preds = apply_threshold(scores, threshold=0.5, higher_is_same=True)
        np.testing.assert_array_equal(preds, [1])

    def test_empty_input(self):
        scores = np.array([])
        preds = apply_threshold(scores, threshold=0.5)
        assert len(preds) == 0


class TestConfusionMatrix:
    def test_perfect_predictions(self):
        labels = np.array([1, 1, 0, 0])
        preds = np.array([1, 1, 0, 0])
        cm = confusion_matrix(labels, preds)
        assert cm == {"tp": 2, "fp": 0, "tn": 2, "fn": 0}

    def test_all_wrong(self):
        labels = np.array([1, 1, 0, 0])
        preds = np.array([0, 0, 1, 1])
        cm = confusion_matrix(labels, preds)
        assert cm == {"tp": 0, "fp": 2, "tn": 0, "fn": 2}

    def test_mixed(self):
        labels = np.array([1, 0, 1, 0, 1])
        preds = np.array([1, 0, 0, 1, 1])
        cm = confusion_matrix(labels, preds)
        assert cm["tp"] == 2
        assert cm["fn"] == 1
        assert cm["fp"] == 1
        assert cm["tn"] == 1


class TestComputeMetrics:
    def test_perfect_score(self):
        labels = np.array([1, 1, 0, 0])
        preds = np.array([1, 1, 0, 0])
        m = compute_metrics(labels, preds)
        assert m["accuracy"] == 1.0
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0
        assert m["balanced_accuracy"] == 1.0

    def test_all_positive_predictions(self):
        labels = np.array([1, 0, 1, 0])
        preds = np.array([1, 1, 1, 1])
        m = compute_metrics(labels, preds)
        assert m["accuracy"] == 0.5
        assert m["recall"] == 1.0
        assert m["precision"] == 0.5

    def test_zero_division_safety(self):
        labels = np.array([0, 0, 0])
        preds = np.array([0, 0, 0])
        m = compute_metrics(labels, preds)
        assert m["accuracy"] == 1.0
        assert m["precision"] == 0.0  # no positive predictions
        assert m["recall"] == 0.0     # no actual positives

    def test_known_f1(self):
        # 2 TP, 1 FP, 1 FN => precision=2/3, recall=2/3, F1=2/3
        labels = np.array([1, 1, 1, 0])
        preds = np.array([1, 1, 0, 1])
        m = compute_metrics(labels, preds)
        np.testing.assert_almost_equal(m["precision"], 2 / 3)
        np.testing.assert_almost_equal(m["recall"], 2 / 3)
        np.testing.assert_almost_equal(m["f1"], 2 / 3)


class TestThresholdSweep:
    def test_sweep_returns_correct_count(self):
        labels = np.array([1, 1, 0, 0])
        scores = np.array([0.9, 0.6, 0.4, 0.1])
        thresholds = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        results = threshold_sweep(labels, scores, thresholds)
        assert len(results) == 5
        assert all("threshold" in r for r in results)
        assert all("f1" in r for r in results)

    def test_extreme_thresholds(self):
        labels = np.array([1, 0])
        scores = np.array([0.8, 0.2])
        results = threshold_sweep(labels, scores, np.array([0.0, 1.0]))
        # threshold=0.0: everything predicted 1 => TP=1, FP=1
        assert results[0]["tp"] == 1
        assert results[0]["fp"] == 1
        # threshold=1.0: only score>=1.0 => nothing predicted 1
        assert results[1]["tp"] == 0
        assert results[1]["fp"] == 0


class TestROCCurveData:
    def test_roc_output_shape(self):
        labels = np.array([1, 1, 0, 0])
        scores = np.array([0.9, 0.7, 0.3, 0.1])
        thresholds = np.linspace(0, 1, 11)
        roc = roc_curve_data(labels, scores, thresholds)
        assert len(roc["fpr"]) == 11
        assert len(roc["tpr"]) == 11
        assert len(roc["thresholds"]) == 11

    def test_roc_boundary_values(self):
        labels = np.array([1, 0])
        scores = np.array([0.9, 0.1])
        roc = roc_curve_data(labels, scores, np.array([0.0, 1.0]))
        # threshold=0 => predict all 1 => TPR=1, FPR=1
        assert roc["tpr"][0] == 1.0
        assert roc["fpr"][0] == 1.0


class TestSelectThreshold:
    def test_max_f1(self):
        sweep = [
            {"threshold": 0.3, "f1": 0.6, "balanced_accuracy": 0.7, "fp": 2, "tn": 3, "tp": 3, "fn": 2},
            {"threshold": 0.5, "f1": 0.9, "balanced_accuracy": 0.8, "fp": 1, "tn": 4, "tp": 4, "fn": 1},
            {"threshold": 0.7, "f1": 0.7, "balanced_accuracy": 0.85, "fp": 0, "tn": 5, "tp": 3, "fn": 2},
        ]
        best = select_threshold(sweep, rule="max_f1")
        assert best["threshold"] == 0.5

    def test_max_balanced_accuracy(self):
        sweep = [
            {"threshold": 0.3, "f1": 0.6, "balanced_accuracy": 0.7, "fp": 2, "tn": 3, "tp": 3, "fn": 2},
            {"threshold": 0.5, "f1": 0.9, "balanced_accuracy": 0.8, "fp": 1, "tn": 4, "tp": 4, "fn": 1},
            {"threshold": 0.7, "f1": 0.7, "balanced_accuracy": 0.85, "fp": 0, "tn": 5, "tp": 3, "fn": 2},
        ]
        best = select_threshold(sweep, rule="max_balanced_accuracy")
        assert best["threshold"] == 0.7

    def test_invalid_rule_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            select_threshold([], rule="nonexistent")
