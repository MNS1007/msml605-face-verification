"""Unit tests for the validation module."""

import numpy as np
import pandas as pd
import pytest

from src.validation import (
    validate_labels,
    validate_metrics_complete,
    validate_no_split_leakage,
    validate_pair_schema,
    validate_score_count,
    validate_split_names,
    validate_threshold_range,
)


def _make_pairs_df(**overrides):
    """Helper to create a valid pairs DataFrame."""
    data = {
        "left_path": ["/a/img1.jpg", "/a/img2.jpg"],
        "right_path": ["/b/img1.jpg", "/b/img2.jpg"],
        "label": [1, 0],
        "split": ["val", "val"],
    }
    data.update(overrides)
    return pd.DataFrame(data)


class TestValidatePairSchema:
    def test_valid_schema(self):
        df = _make_pairs_df()
        validate_pair_schema(df)  # should not raise

    def test_missing_column(self):
        df = _make_pairs_df()
        df = df.drop(columns=["label"])
        with pytest.raises(ValueError, match="missing required columns"):
            validate_pair_schema(df)


class TestValidateLabels:
    def test_valid_labels(self):
        df = _make_pairs_df()
        validate_labels(df)  # should not raise

    def test_invalid_label(self):
        df = _make_pairs_df(label=[1, 2])
        with pytest.raises(ValueError, match="Invalid label"):
            validate_labels(df)


class TestValidateSplitNames:
    def test_valid_splits(self):
        df = _make_pairs_df(split=["train", "test"])
        validate_split_names(df)

    def test_invalid_split(self):
        df = _make_pairs_df(split=["train", "unknown"])
        with pytest.raises(ValueError, match="Invalid split"):
            validate_split_names(df)


class TestValidateScoreCount:
    def test_matching_count(self):
        validate_score_count(np.array([0.5, 0.6]), expected_n=2)

    def test_mismatched_count(self):
        with pytest.raises(ValueError, match="Score count mismatch"):
            validate_score_count(np.array([0.5, 0.6, 0.7]), expected_n=2)


class TestValidateThresholdRange:
    def test_valid_threshold(self):
        validate_threshold_range(0.5, min_val=0.0, max_val=1.0)

    def test_below_range(self):
        with pytest.raises(ValueError, match="outside allowed range"):
            validate_threshold_range(-0.1, min_val=0.0, max_val=1.0)

    def test_above_range(self):
        with pytest.raises(ValueError, match="outside allowed range"):
            validate_threshold_range(1.5, min_val=0.0, max_val=1.0)

    def test_cosine_range(self):
        validate_threshold_range(-0.5, min_val=-1.0, max_val=1.0)


class TestValidateNoSplitLeakage:
    def test_no_leakage(self):
        val_df = _make_pairs_df()
        test_df = pd.DataFrame({
            "left_path": ["/c/img1.jpg"],
            "right_path": ["/d/img1.jpg"],
            "label": [1],
            "split": ["test"],
        })
        validate_no_split_leakage(val_df, test_df)

    def test_leakage_detected(self):
        val_df = _make_pairs_df()
        test_df = val_df.copy()
        test_df["split"] = "test"
        with pytest.raises(ValueError, match="Split leakage"):
            validate_no_split_leakage(val_df, test_df)


class TestValidateMetricsComplete:
    def test_valid_metrics(self):
        metrics = {
            "accuracy": 0.9, "precision": 0.8, "recall": 0.85,
            "f1": 0.82, "tp": 40, "fp": 10, "tn": 50, "fn": 7,
        }
        validate_metrics_complete(metrics)

    def test_missing_key(self):
        metrics = {"accuracy": 0.9, "tp": 40, "fp": 10, "tn": 50, "fn": 7}
        with pytest.raises(ValueError, match="missing required keys"):
            validate_metrics_complete(metrics)

    def test_zero_total(self):
        metrics = {
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0,
            "f1": 0.0, "tp": 0, "fp": 0, "tn": 0, "fn": 0,
        }
        with pytest.raises(ValueError, match="zero total"):
            validate_metrics_complete(metrics)
