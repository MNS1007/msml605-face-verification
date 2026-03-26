"""Unit tests for the data-centric improvement module."""

import numpy as np
import pandas as pd
import pytest

from src.data_centric import (
    cap_overrepresented_identities,
    count_identity_appearances,
    rebalance_labels,
)


def _make_toy_pairs():
    """Create a small pairs DataFrame with known identity distribution.

    Identities:
    - Alice: appears in 5 pairs (overrepresented)
    - Bob: appears in 2 pairs
    - Carol: appears in 1 pair
    """
    return pd.DataFrame({
        "left_path": [
            "/data/Alice/Alice_0001.jpg",
            "/data/Alice/Alice_0001.jpg",
            "/data/Alice/Alice_0002.jpg",
            "/data/Alice/Alice_0002.jpg",
            "/data/Bob/Bob_0001.jpg",
            "/data/Carol/Carol_0001.jpg",
        ],
        "right_path": [
            "/data/Alice/Alice_0002.jpg",
            "/data/Bob/Bob_0001.jpg",
            "/data/Carol/Carol_0001.jpg",
            "/data/Bob/Bob_0002.jpg",
            "/data/Bob/Bob_0002.jpg",
            "/data/Alice/Alice_0003.jpg",
        ],
        "label": [1, 0, 0, 0, 1, 0],
        "split": ["val"] * 6,
    })


class TestCountIdentityAppearances:
    def test_counts(self):
        df = _make_toy_pairs()
        counts = count_identity_appearances(df)
        assert counts["Alice"] == 6  # appears in left or right of 5 pairs, plus once more in right of pair 6
        assert counts["Bob"] == 4    # pairs 2,4,5 left+right


class TestCapOverrepresentedIdentities:
    def test_cap_reduces_pairs(self):
        df = _make_toy_pairs()
        original_len = len(df)
        capped = cap_overrepresented_identities(df, max_pairs_per_identity=3, seed=42)
        # Capping should remove at least one pair
        assert len(capped) < original_len
        # The cap limits row participation: count how many rows each identity
        # participates in (appears in left_path OR right_path).
        from src.data_centric import _identity_from_path
        for name in ["Alice", "Bob", "Carol"]:
            rows_with = capped[
                capped["left_path"].apply(_identity_from_path).eq(name)
                | capped["right_path"].apply(_identity_from_path).eq(name)
            ]
            assert len(rows_with) <= 3, f"{name} in {len(rows_with)} pairs after cap=3"

    def test_cap_is_deterministic(self):
        df = _make_toy_pairs()
        r1 = cap_overrepresented_identities(df, max_pairs_per_identity=3, seed=42)
        r2 = cap_overrepresented_identities(df, max_pairs_per_identity=3, seed=42)
        pd.testing.assert_frame_equal(r1, r2)

    def test_high_cap_no_change(self):
        df = _make_toy_pairs()
        capped = cap_overrepresented_identities(df, max_pairs_per_identity=100, seed=42)
        assert len(capped) == len(df)


class TestRebalanceLabels:
    def test_balanced_output(self):
        df = pd.DataFrame({
            "left_path": [f"/a/{i}.jpg" for i in range(10)],
            "right_path": [f"/b/{i}.jpg" for i in range(10)],
            "label": [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],  # 7 pos, 3 neg
            "split": ["val"] * 10,
        })
        balanced = rebalance_labels(df, seed=42)
        pos = (balanced["label"] == 1).sum()
        neg = (balanced["label"] == 0).sum()
        assert pos == neg == 3  # minority is 3

    def test_already_balanced(self):
        df = pd.DataFrame({
            "left_path": ["/a/1.jpg", "/a/2.jpg"],
            "right_path": ["/b/1.jpg", "/b/2.jpg"],
            "label": [1, 0],
            "split": ["val", "val"],
        })
        balanced = rebalance_labels(df, seed=42)
        assert len(balanced) == 2

    def test_deterministic(self):
        df = pd.DataFrame({
            "left_path": [f"/a/{i}.jpg" for i in range(8)],
            "right_path": [f"/b/{i}.jpg" for i in range(8)],
            "label": [1, 1, 1, 1, 1, 0, 0, 0],
            "split": ["val"] * 8,
        })
        r1 = rebalance_labels(df, seed=42)
        r2 = rebalance_labels(df, seed=42)
        pd.testing.assert_frame_equal(r1, r2)
