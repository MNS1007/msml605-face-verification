"""Data-centric improvement utilities for the face verification pipeline.

Milestone 2 requires at least one meaningful change to the data, pair
construction, filtering, or balancing process.  This module implements:

1. Identity-cap filtering — limits the maximum number of pairs any single
   identity can participate in, preventing a few heavily-photographed people
   from dominating the evaluation.

2. Missing-image filtering — removes pairs whose image files are not found
   on disk, so downstream scoring never fails silently.

3. Positive/negative rebalancing — ensures a controlled 1:1 ratio of
   same-identity and different-identity pairs per split for fair evaluation.

What changed and why
--------------------
*Problem*: In the original LFW pair set, some identities (e.g. George_W_Bush,
Colin_Powell) appear in far more pairs than others.  This skews aggregate
metrics — the verifier can look good by getting a few popular identities right
while silently failing on rare ones.

*Change*: We cap each identity to at most ``max_pairs_per_identity`` pair
appearances (default 30), drop pairs referencing missing images, and then
downsample the majority class to restore a 1:1 label balance.  All operations
are deterministic (sorted, seeded).

*Effect*: The evaluation set becomes more representative of the long tail of
identities, giving a fairer picture of verifier robustness.
"""

import os

import numpy as np
import pandas as pd


def _identity_from_path(path: str) -> str:
    """Extract the identity name from an LFW image path."""
    basename = os.path.basename(os.path.dirname(path))
    return basename


def count_identity_appearances(df: pd.DataFrame) -> pd.Series:
    """Count how many pairs each identity appears in (left or right)."""
    left_ids = df["left_path"].apply(_identity_from_path)
    right_ids = df["right_path"].apply(_identity_from_path)
    all_ids = pd.concat([left_ids, right_ids])
    return all_ids.value_counts()


def cap_overrepresented_identities(df: pd.DataFrame,
                                   max_pairs_per_identity: int = 30,
                                   seed: int = 42) -> pd.DataFrame:
    """Remove pairs so no identity appears in more than max_pairs_per_identity pairs.

    The function iteratively drops pairs from the most overrepresented
    identities until the cap is satisfied.  The process is deterministic.

    Parameters
    ----------
    df : pairs DataFrame with columns left_path, right_path, label, split.
    max_pairs_per_identity : maximum pair appearances per identity.
    seed : random seed for reproducible sampling when dropping.

    Returns
    -------
    Filtered DataFrame (copy).
    """
    df = df.copy().reset_index(drop=True)
    rng = np.random.default_rng(seed)

    left_ids = df["left_path"].apply(_identity_from_path)
    right_ids = df["right_path"].apply(_identity_from_path)

    # Build a mapping: identity -> set of row indices it appears in
    identity_rows: dict[str, set[int]] = {}
    for idx in range(len(df)):
        for name in [left_ids.iloc[idx], right_ids.iloc[idx]]:
            identity_rows.setdefault(name, set()).add(idx)

    rows_to_drop: set[int] = set()
    for name, row_set in sorted(identity_rows.items()):
        active = row_set - rows_to_drop
        if len(active) > max_pairs_per_identity:
            excess = list(active)
            rng.shuffle(excess)
            n_drop = len(active) - max_pairs_per_identity
            rows_to_drop.update(excess[:n_drop])

    result = df.drop(index=list(rows_to_drop)).reset_index(drop=True)
    return result


def filter_missing_images(df: pd.DataFrame) -> pd.DataFrame:
    """Remove pairs where either image file does not exist on disk."""
    mask = df.apply(
        lambda row: os.path.isfile(row["left_path"]) and os.path.isfile(row["right_path"]),
        axis=1,
    )
    return df[mask].reset_index(drop=True)


def rebalance_labels(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Downsample the majority label class to match the minority class count.

    This gives a 1:1 positive-to-negative ratio for fair metric computation.
    """
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]
    n_minority = min(len(pos), len(neg))
    pos_sampled = pos.sample(n=n_minority, random_state=seed).reset_index(drop=True)
    neg_sampled = neg.sample(n=n_minority, random_state=seed).reset_index(drop=True)
    result = pd.concat([pos_sampled, neg_sampled], ignore_index=True)
    result = result.sort_values(by=["left_path", "right_path"]).reset_index(drop=True)
    return result


def apply_all_improvements(df: pd.DataFrame, max_pairs_per_identity: int = 30,
                           seed: int = 42, check_images: bool = True) -> pd.DataFrame:
    """Apply the full data-centric improvement pipeline.

    Steps (in order):
    1. Filter missing images (if check_images=True).
    2. Cap overrepresented identities.
    3. Rebalance positive/negative labels.

    Returns
    -------
    Improved DataFrame with a summary printed to stdout.
    """
    original_count = len(df)

    if check_images:
        df = filter_missing_images(df)
        after_image_filter = len(df)
    else:
        after_image_filter = original_count

    df = cap_overrepresented_identities(df, max_pairs_per_identity, seed)
    after_cap = len(df)

    df = rebalance_labels(df, seed)
    after_balance = len(df)

    print(f"Data-centric improvement summary:")
    print(f"  Original pairs:          {original_count}")
    print(f"  After image filter:      {after_image_filter} (removed {original_count - after_image_filter})")
    print(f"  After identity cap ({max_pairs_per_identity}):  {after_cap} (removed {after_image_filter - after_cap})")
    print(f"  After label rebalance:   {after_balance} (removed {after_cap - after_balance})")
    pos = (df["label"] == 1).sum()
    neg = (df["label"] == 0).sum()
    print(f"  Final: {pos} positive, {neg} negative")

    return df
