import os
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.tracking.run_logger import create_run, log_threshold, append_run_csv


def confusion_counts(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, fp, tn, fn

# These functions are for the confusion matrix
def confusion_matrix_2x2(tp, fp, tn, fn):
    return [[tn, fp], [fn, tp]]


def compute_metrics(tp, fp, tn, fn):
    total = tp + fp + tn + fn

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return accuracy, precision, recall, f1

def find_best_threshold(scores, labels, thresholds):
    # Finds the best threshold and returns them
    best_threshold = None
    best_f1 = -1
    best_metrics = None
    best_confusion = None

    for t in thresholds:
        preds = (scores >= t).astype(int)
        tp, fp, tn, fn = confusion_counts(labels, preds)
        acc, prec, rec, f1 = compute_metrics(tp, fp, tn, fn)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)
            best_metrics = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
            }
            best_confusion = {
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "matrix_2x2": confusion_matrix_2x2(tp, fp, tn, fn),
            }

    return best_threshold, best_metrics, best_confusion


def main():
    run_id, run_dir, run_info = create_run(
        note="Run 5: threshold selection after filtering invalid image paths",
        data_version="filtered_pairs_v1"
    )
    val_path = os.path.join("outputs", "scores", "val_scored.csv")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Missing file: {val_path}. Run score_pairs.py first.")

    df = pd.read_csv(val_path)

    if "score" not in df.columns or "label" not in df.columns:
        raise ValueError("val_scored.csv must contain columns: score, label")

    scores = df["score"].to_numpy()
    labels = df["label"].to_numpy().astype(int)

    thresholds = np.linspace(-0.5, 1.0, 200)

    print(f"Selecting best threshold on validation set (N={len(df)})...")
    best_threshold, best_metrics, best_confusion = find_best_threshold(scores, labels, thresholds)
    


    print("\nBest threshold selected (max F1):")
    print(f"  threshold = {best_threshold:.6f}")
    print(f"  accuracy  = {best_metrics['accuracy']:.4f}")
    print(f"  precision = {best_metrics['precision']:.4f}")
    print(f"  recall    = {best_metrics['recall']:.4f}")
    print(f"  f1        = {best_metrics['f1']:.4f}")

    print("\nConfusion matrix counts:")
    print(f"  TP = {best_confusion['tp']}")
    print(f"  FP = {best_confusion['fp']}")
    print(f"  TN = {best_confusion['tn']}")
    print(f"  FN = {best_confusion['fn']}")

    print("\nConfusion matrix 2x2 format [[TN, FP], [FN, TP]]:")
    print(best_confusion["matrix_2x2"])

    # Save artifacts
    out_dir = os.path.join("outputs", "threshold")
    os.makedirs(out_dir, exist_ok=True)

    threshold_path = os.path.join(out_dir, "selected_threshold.json")
    confusion_path = os.path.join(out_dir, "val_confusion_matrix.json")

    threshold_payload = {
        "selection_rule": "max_f1_on_validation",
        "threshold": best_threshold,
        "metrics": best_metrics,
    }
    log_threshold(run_dir, threshold_payload)
    append_run_csv(
        run_info,
        metrics=best_metrics,
        threshold=best_threshold
    )
    with open(threshold_path, "w") as f:
        json.dump(threshold_payload, f, indent=2)

    with open(confusion_path, "w") as f:
        json.dump(best_confusion, f, indent=2)

    print(f"\nSaved selected threshold to: {threshold_path}")
    print(f"Saved validation confusion matrix to: {confusion_path}")


if __name__ == "__main__":
    main()