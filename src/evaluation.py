"""Evaluation utilities for the face verification pipeline.

Provides metrics computation, threshold application, confusion matrix
construction, and ROC curve data generation.
"""

import numpy as np


def apply_threshold(scores: np.ndarray, threshold: float, higher_is_same: bool = True) -> np.ndarray:
    """Convert continuous scores to binary predictions using a threshold.

    Parameters
    ----------
    scores : array of shape (N,)
        Similarity scores for each pair.
    threshold : float
        Decision boundary.
    higher_is_same : bool
        If True, scores >= threshold predict "same". If False, scores <= threshold predict "same".

    Returns
    -------
    predictions : array of shape (N,) with values 0 or 1.
    """
    scores = np.asarray(scores, dtype=float)
    if higher_is_same:
        return (scores >= threshold).astype(int)
    return (scores <= threshold).astype(int)


def confusion_matrix(labels: np.ndarray, predictions: np.ndarray):
    """Compute TP, FP, TN, FN counts.

    Parameters
    ----------
    labels : array of 0/1 ground-truth labels.
    predictions : array of 0/1 predicted labels.

    Returns
    -------
    dict with keys 'tp', 'fp', 'tn', 'fn'.
    """
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    tp = int(np.sum((predictions == 1) & (labels == 1)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def compute_metrics(labels: np.ndarray, predictions: np.ndarray):
    """Compute accuracy, precision, recall, F1, and balanced accuracy.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, balanced_accuracy,
                    tp, fp, tn, fn.
    """
    cm = confusion_matrix(labels, predictions)
    tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
    total = tp + fp + tn + fn

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    tpr = recall
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_accuracy = (tpr + tnr) / 2.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        **cm,
    }


def threshold_sweep(labels: np.ndarray, scores: np.ndarray,
                    thresholds: np.ndarray, higher_is_same: bool = True):
    """Evaluate metrics across a range of thresholds.

    Returns
    -------
    list of dicts, one per threshold, each containing 'threshold' and all
    keys from compute_metrics.
    """
    results = []
    for t in thresholds:
        preds = apply_threshold(scores, t, higher_is_same=higher_is_same)
        m = compute_metrics(labels, preds)
        m["threshold"] = float(t)
        results.append(m)
    return results


def roc_curve_data(labels: np.ndarray, scores: np.ndarray,
                   thresholds: np.ndarray, higher_is_same: bool = True):
    """Compute FPR and TPR arrays for an ROC-style plot.

    Returns
    -------
    dict with keys 'fpr', 'tpr', 'thresholds' — each a list of floats.
    """
    labels = np.asarray(labels, dtype=int)
    fprs, tprs = [], []
    for t in thresholds:
        preds = apply_threshold(scores, t, higher_is_same=higher_is_same)
        cm = confusion_matrix(labels, preds)
        tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fprs.append(fpr)
        tprs.append(tpr)
    return {"fpr": fprs, "tpr": tprs, "thresholds": [float(t) for t in thresholds]}


def select_threshold(sweep_results, rule="max_f1"):
    """Select the best threshold from sweep results using a stated rule.

    Supported rules
    ---------------
    - 'max_f1': threshold that maximises F1 score.
    - 'max_balanced_accuracy': threshold that maximises balanced accuracy.
    - 'eer': threshold closest to equal error rate (FPR ≈ FNR).

    Returns
    -------
    dict — the sweep result entry for the selected threshold.
    """
    if rule == "max_f1":
        return max(sweep_results, key=lambda r: r["f1"])
    elif rule == "max_balanced_accuracy":
        return max(sweep_results, key=lambda r: r["balanced_accuracy"])
    elif rule == "eer":
        def eer_distance(r):
            total_neg = r["fp"] + r["tn"]
            total_pos = r["tp"] + r["fn"]
            fpr = r["fp"] / total_neg if total_neg > 0 else 0
            fnr = r["fn"] / total_pos if total_pos > 0 else 0
            return abs(fpr - fnr)
        return min(sweep_results, key=eer_distance)
    else:
        raise ValueError(f"Unknown threshold selection rule: {rule}")
