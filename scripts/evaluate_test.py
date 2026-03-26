import os
import json
import pandas as pd
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.tracking.run_logger import create_run, log_metrics, append_run_csv

# These functions are to create the confusion matrix
def confusion_counts(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, fp, tn, fn


def confusion_matrix_2x2(tp, fp, tn, fn):
    return [[tn, fp], [fn, tp]]


def compute_metrics(tp, fp, tn, fn):
    total = tp + fp + tn + fn

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return accuracy, precision, recall, f1



def main():
    # This adds the tracked run, update it for whatever version it is
    run_id, run_dir, run_info = create_run(
        note="Run 6: test evaluation after filtering invalid image paths",
        data_version="filtered_pairs_v1"
    )

    threshold_path = os.path.join("outputs", "threshold", "selected_threshold.json")
    test_path = os.path.join("outputs", "scores", "test_scored.csv")

    if not os.path.exists(threshold_path):
        raise FileNotFoundError(f"Missing threshold file: {threshold_path}. Run select_threshold.py first.")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing test scored file: {test_path}. Run score_pairs.py first.")

    # Load threshold
    with open(threshold_path, "r") as f:
        threshold_data = json.load(f)

    threshold = threshold_data["threshold"]
    print(f"Loaded selected threshold: {threshold:.6f}")

    # Load test data
    df = pd.read_csv(test_path)

    if "score" not in df.columns or "label" not in df.columns:
        raise ValueError("test_scored.csv must contain columns: score, label")

    scores = df["score"]
    labels = df["label"].astype(int)
    preds = (scores >= threshold).astype(int)

    # Compute metrics
    tp, fp, tn, fn = confusion_counts(labels, preds)
    acc, prec, rec, f1 = compute_metrics(tp, fp, tn, fn)

    metrics = {
        "split": "test",
        "threshold": threshold,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }

    confusion = {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "matrix_2x2": confusion_matrix_2x2(tp, fp, tn, fn),
    }

    print("\nTEST RESULTS")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1       : {f1:.4f}")

    print("\nConfusion matrix counts:")
    print(f"  TP = {tp}")
    print(f"  FP = {fp}")
    print(f"  TN = {tn}")
    print(f"  FN = {fn}")

    out_dir = os.path.join("outputs", "eval")
    os.makedirs(out_dir, exist_ok=True)

    metrics_path = os.path.join(out_dir, "test_metrics.json")
    confusion_path = os.path.join(out_dir, "test_confusion_matrix.json")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    with open(confusion_path, "w") as f:
        json.dump(confusion, f, indent=2)

    log_metrics(run_id, run_dir, metrics)

    append_run_csv(
        run_info,
        metrics=metrics,
        threshold=threshold
    )

    print(f"\nSaved test metrics to: {metrics_path}")
    print(f"Saved test confusion matrix to: {confusion_path}")


if __name__ == "__main__":
    main()