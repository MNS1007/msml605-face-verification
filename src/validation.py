"""Pipeline validation checks for the face verification system.

Each function raises ValueError with a descriptive message on failure,
making the pipeline fail early when inputs or outputs are malformed.
"""

import os

import numpy as np
import pandas as pd


REQUIRED_PAIR_COLUMNS = {"left_path", "right_path", "label", "split"}
VALID_LABELS = {0, 1}
VALID_SPLITS = {"train", "val", "test"}


def validate_pair_schema(df: pd.DataFrame) -> None:
    """Check that a pairs DataFrame has the required columns."""
    missing = REQUIRED_PAIR_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Pair file missing required columns: {missing}")


def validate_labels(df: pd.DataFrame) -> None:
    """Check that all labels are valid binary values (0 or 1)."""
    unique_labels = set(df["label"].unique())
    invalid = unique_labels - VALID_LABELS
    if invalid:
        raise ValueError(f"Invalid label values found: {invalid}. Expected only {VALID_LABELS}")


def validate_split_names(df: pd.DataFrame) -> None:
    """Check that all split names are valid."""
    unique_splits = set(df["split"].unique())
    invalid = unique_splits - VALID_SPLITS
    if invalid:
        raise ValueError(f"Invalid split names found: {invalid}. Expected only {VALID_SPLITS}")


def validate_image_paths_exist(df: pd.DataFrame, sample_n: int = 0) -> None:
    """Check that referenced image paths exist on disk.

    Parameters
    ----------
    df : pairs DataFrame
    sample_n : if > 0, only check a random sample of this many rows for speed.
               if 0, check all rows.
    """
    check_df = df if sample_n == 0 else df.sample(n=min(sample_n, len(df)), random_state=42)
    missing = []
    for _, row in check_df.iterrows():
        for col in ["left_path", "right_path"]:
            path = row[col]
            if not os.path.isfile(path):
                missing.append(path)
    if missing:
        raise ValueError(
            f"{len(missing)} image path(s) not found. First few: {missing[:5]}"
        )


def validate_score_count(scores: np.ndarray, expected_n: int) -> None:
    """Check that the number of scores matches the number of evaluated pairs."""
    if len(scores) != expected_n:
        raise ValueError(
            f"Score count mismatch: got {len(scores)} scores for {expected_n} pairs"
        )


def validate_threshold_range(threshold: float, min_val: float = -1.0,
                             max_val: float = 1.0) -> None:
    """Check that a threshold is within the allowed numeric range."""
    if not (min_val <= threshold <= max_val):
        raise ValueError(
            f"Threshold {threshold} is outside allowed range [{min_val}, {max_val}]"
        )


def validate_no_split_leakage(val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Check that no pair appears in both validation and test sets."""
    val_keys = set(zip(val_df["left_path"], val_df["right_path"]))
    test_keys = set(zip(test_df["left_path"], test_df["right_path"]))
    overlap = val_keys & test_keys
    if overlap:
        raise ValueError(
            f"Split leakage detected: {len(overlap)} pair(s) appear in both val and test"
        )


def validate_metrics_complete(metrics: dict) -> None:
    """Check that logged metrics are not missing and denominators are valid."""
    required_keys = {"accuracy", "precision", "recall", "f1", "tp", "fp", "tn", "fn"}
    missing = required_keys - set(metrics.keys())
    if missing:
        raise ValueError(f"Metrics missing required keys: {missing}")
    total = metrics["tp"] + metrics["fp"] + metrics["tn"] + metrics["fn"]
    if total == 0:
        raise ValueError("Metrics have zero total (tp+fp+tn+fn=0), invalid evaluation")


def validate_pairs_file(filepath: str) -> pd.DataFrame:
    """Load and fully validate a pairs CSV file. Returns the DataFrame if valid."""
    df = pd.read_csv(filepath)
    validate_pair_schema(df)
    validate_labels(df)
    validate_split_names(df)
    return df
