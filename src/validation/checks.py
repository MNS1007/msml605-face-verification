import os
import pandas as pd


def validate_schema(df: pd.DataFrame, split_name="unknown"):
    required_cols = ["left_path", "right_path", "label", "split"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"[{split_name}] Missing required column: {col}")

    print(f"[{split_name}] Schema validation passed")



def validate_labels(df: pd.DataFrame, split_name="unknown"):
    unique_labels = set(df["label"].unique())

    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"[{split_name}] Invalid labels found: {unique_labels}")

    print(f"[{split_name}] Label validation passed")


def validate_image_paths(df: pd.DataFrame, split_name="unknown", max_checks=100):
    """
    Only checks first `max_checks` rows for speed.
    """
    sample_df = df.head(max_checks)

    for i, row in sample_df.iterrows():
        if not os.path.exists(row["left_path"]):
            raise FileNotFoundError(f"[{split_name}] Missing image: {row['left_path']}")

        if not os.path.exists(row["right_path"]):
            raise FileNotFoundError(f"[{split_name}] Missing image: {row['right_path']}")

    print(f"[{split_name}] Image path validation passed (checked {len(sample_df)} samples)")

def validate_score_count(pairs_df: pd.DataFrame, scored_df: pd.DataFrame, split_name="unknown"):
    if len(pairs_df) != len(scored_df):
        raise ValueError(
            f"[{split_name}] Score count mismatch: pairs={len(pairs_df)} vs scores={len(scored_df)}"
        )

    print(f"[{split_name}] Score count validation passed")

def validate_no_leakage(train_df, val_df, test_df):
    def pair_set(df):
        return set(zip(df["left_path"], df["right_path"]))

    train_pairs = pair_set(train_df)
    val_pairs = pair_set(val_df)
    test_pairs = pair_set(test_df)

    if train_pairs & val_pairs:
        raise ValueError("Leakage detected between train and val splits")

    if train_pairs & test_pairs:
        raise ValueError("Leakage detected between train and test splits")

    if val_pairs & test_pairs:
        raise ValueError("Leakage detected between val and test splits")

    print("[ALL SPLITS] No leakage detected")

def validate_threshold(threshold, metric="cosine"):
    if metric == "cosine":
        if not (-1 <= threshold <= 1):
            raise ValueError(f"Invalid cosine threshold: {threshold}")

    elif metric == "euclidean":
        if threshold < 0:
            raise ValueError(f"Invalid euclidean threshold: {threshold}")

    else:
        raise ValueError(f"Unknown metric type: {metric}")

    print("Threshold validation passed")


def validate_all(
    train_pairs_path,
    val_pairs_path,
    test_pairs_path,
    val_scored_path=None,
    test_scored_path=None,
    threshold=None,
):
    print("\n=== Running Validation Checks ===")

    train_df = pd.read_csv(train_pairs_path)
    val_df = pd.read_csv(val_pairs_path)
    test_df = pd.read_csv(test_pairs_path)

    # Schema + labels
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        validate_schema(df, name)
        validate_labels(df, name)
        validate_image_paths(df, name)

    # Leakage
    validate_no_leakage(train_df, val_df, test_df)

    # Score checks
    if val_scored_path:
        val_scored_df = pd.read_csv(val_scored_path)
        validate_score_count(val_df, val_scored_df, "val")

    if test_scored_path:
        test_scored_df = pd.read_csv(test_scored_path)
        validate_score_count(test_df, test_scored_df, "test")

    # Threshold check
    if threshold is not None:
        validate_threshold(threshold)

    print("\nAll validation checks passed")
