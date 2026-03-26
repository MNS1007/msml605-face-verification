"""Error analysis utilities for the face verification pipeline.

Implements two error slices as required by Milestone 2:

Slice 1 — False negatives among same-identity pairs with rare identities
    Definition: Same-person pairs (label=1) that the verifier incorrectly
    rejects, restricted to identities that have very few images in the
    dataset (≤ a configurable threshold).
    Hypothesis: Identities with few training images produce less stable
    embeddings, leading to higher intra-identity variance and missed matches.

Slice 2 — False positives near the decision boundary
    Definition: Different-person pairs (label=0) that the verifier
    incorrectly accepts, restricted to pairs whose similarity score falls
    within a narrow band above the decision threshold.
    Hypothesis: These pairs involve visually similar faces (similar age,
    gender, ethnicity, or pose) whose embeddings land in nearby regions
    of feature space, making them hard to separate at any single threshold.
"""

import os

import numpy as np
import pandas as pd


def _identity_from_path(path: str) -> str:
    return os.path.basename(os.path.dirname(path))


def _count_images_per_identity(df: pd.DataFrame) -> dict[str, int]:
    """Count unique images per identity across all pairs."""
    images: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        for col in ["left_path", "right_path"]:
            path = row[col]
            name = _identity_from_path(path)
            images.setdefault(name, set()).add(path)
    return {name: len(paths) for name, paths in images.items()}


def slice_false_negatives_rare_identities(
    df: pd.DataFrame,
    scores: np.ndarray,
    predictions: np.ndarray,
    max_images_per_identity: int = 4,
):
    """Slice 1: False negatives among same-identity pairs with rare identities.

    Parameters
    ----------
    df : pairs DataFrame.
    scores : similarity scores array.
    predictions : binary prediction array.
    max_images_per_identity : identities with at most this many unique images
        in the dataset are considered "rare".

    Returns
    -------
    dict with keys:
        - 'slice_name': str
        - 'definition': str
        - 'total_in_slice': int (same-identity pairs with rare identities)
        - 'errors_in_slice': int (false negatives in this slice)
        - 'error_rate': float
        - 'examples': list of dicts (up to 5 representative examples)
        - 'hypothesis': str
        - 'future_improvement': str
    """
    labels = df["label"].values
    image_counts = _count_images_per_identity(df)

    # Identify rare identities
    rare_ids = {name for name, count in image_counts.items() if count <= max_images_per_identity}

    # Find same-identity pairs involving at least one rare identity
    slice_mask = np.zeros(len(df), dtype=bool)
    for idx, row in df.iterrows():
        if labels[idx] != 1:
            continue
        left_id = _identity_from_path(row["left_path"])
        right_id = _identity_from_path(row["right_path"])
        if left_id in rare_ids or right_id in rare_ids:
            slice_mask[idx] = True

    # False negatives within this slice
    fn_mask = slice_mask & (predictions == 0) & (labels == 1)

    total_in_slice = int(slice_mask.sum())
    errors_in_slice = int(fn_mask.sum())
    error_rate = errors_in_slice / total_in_slice if total_in_slice > 0 else 0.0

    # Collect representative examples
    fn_indices = np.where(fn_mask)[0]
    example_indices = fn_indices[:5]
    examples = []
    for idx in example_indices:
        row = df.iloc[idx]
        left_id = _identity_from_path(row["left_path"])
        examples.append({
            "pair_index": int(idx),
            "identity": left_id,
            "identity_image_count": image_counts.get(left_id, 0),
            "score": float(scores[idx]),
            "left_image": os.path.basename(row["left_path"]),
            "right_image": os.path.basename(row["right_path"]),
        })

    return {
        "slice_name": "False negatives — rare identities",
        "definition": (
            f"Same-identity pairs (label=1) predicted as different (pred=0), "
            f"where the identity has ≤{max_images_per_identity} unique images in the dataset."
        ),
        "total_in_slice": total_in_slice,
        "errors_in_slice": errors_in_slice,
        "error_rate": error_rate,
        "examples": examples,
        "hypothesis": (
            "Identities with very few images produce less stable embeddings "
            "because there is limited intra-person variation captured. Small "
            "pose or lighting changes cause the embedding to shift enough to "
            "fall below the similarity threshold, resulting in false rejections."
        ),
        "future_improvement": (
            "Augment rare-identity images (flips, slight crops) during pair "
            "construction, or use an embedding model with better few-shot "
            "generalisation. Alternatively, lower the threshold for pairs "
            "flagged as rare-identity to trade precision for recall."
        ),
    }


def slice_false_positives_near_boundary(
    df: pd.DataFrame,
    scores: np.ndarray,
    predictions: np.ndarray,
    threshold: float,
    boundary_width: float = 0.05,
):
    """Slice 2: False positives with scores near the decision boundary.

    Parameters
    ----------
    df : pairs DataFrame.
    scores : similarity scores array.
    predictions : binary prediction array.
    threshold : the operating threshold used.
    boundary_width : width of the band above the threshold to inspect.

    Returns
    -------
    dict with the same structure as slice 1.
    """
    labels = df["label"].values

    # Different-identity pairs predicted as same, with score in [threshold, threshold + width]
    fp_mask = (predictions == 1) & (labels == 0)
    boundary_mask = fp_mask & (scores >= threshold) & (scores < threshold + boundary_width)

    # The full slice is all different-identity pairs in the boundary band
    diff_in_band = (labels == 0) & (scores >= threshold) & (scores < threshold + boundary_width)
    total_in_slice = int(diff_in_band.sum())
    errors_in_slice = int(boundary_mask.sum())
    error_rate = errors_in_slice / total_in_slice if total_in_slice > 0 else 0.0

    # Representative examples
    err_indices = np.where(boundary_mask)[0]
    example_indices = err_indices[:5]
    examples = []
    for idx in example_indices:
        row = df.iloc[idx]
        left_id = _identity_from_path(row["left_path"])
        right_id = _identity_from_path(row["right_path"])
        examples.append({
            "pair_index": int(idx),
            "left_identity": left_id,
            "right_identity": right_id,
            "score": float(scores[idx]),
            "distance_from_threshold": float(scores[idx] - threshold),
            "left_image": os.path.basename(row["left_path"]),
            "right_image": os.path.basename(row["right_path"]),
        })

    return {
        "slice_name": "False positives — near decision boundary",
        "definition": (
            f"Different-identity pairs (label=0) predicted as same (pred=1), "
            f"with similarity score in [{threshold:.4f}, {threshold + boundary_width:.4f})."
        ),
        "total_in_slice": total_in_slice,
        "errors_in_slice": errors_in_slice,
        "error_rate": error_rate,
        "examples": examples,
        "hypothesis": (
            "These false accepts involve visually similar individuals whose "
            "embeddings land close together in feature space — often people of "
            "similar age, gender, and ethnicity photographed under comparable "
            "conditions. The verifier cannot distinguish them at any threshold "
            "near the operating point, so they cluster just above the boundary."
        ),
        "future_improvement": (
            "Use a more discriminative embedding model (e.g. ArcFace) that "
            "pushes different-identity embeddings further apart. Alternatively, "
            "add hard-negative mining during pair construction so the system is "
            "explicitly tested on challenging look-alike pairs."
        ),
    }


def run_error_analysis(df, scores, predictions, threshold, max_images=4,
                       boundary_width=0.05):
    """Run both error slices and return a combined report dict."""
    slice1 = slice_false_negatives_rare_identities(
        df, scores, predictions, max_images_per_identity=max_images,
    )
    slice2 = slice_false_positives_near_boundary(
        df, scores, predictions, threshold=threshold,
        boundary_width=boundary_width,
    )
    return {"slices": [slice1, slice2]}
