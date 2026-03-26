import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.tracking.run_logger import create_run, log_artifact, append_run_csv


def confusion_counts(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, fp, tn, fn


def compute_metrics(tp, fp, tn, fn):
    total = tp + fp + tn + fn

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    tpr = recall
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return accuracy, precision, recall, f1, tpr, fpr


def sweep_thresholds(scores, labels, thresholds):
    rows = []

    for t in thresholds:
        preds = (scores >= t).astype(int)

        tp, fp, tn, fn = confusion_counts(labels, preds)
        acc, prec, rec, f1, tpr, fpr = compute_metrics(tp, fp, tn, fn)

        rows.append({
            "threshold": float(t),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tpr": tpr,
            "fpr": fpr,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn
        })

    return pd.DataFrame(rows)


def plot_roc(df, output_path):
    plt.figure(figsize=(7, 6))
    plt.plot(df["fpr"], df["tpr"], marker="o", markersize=2)
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve (Validation Threshold Sweep)")
    plt.grid(True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    run_id, run_dir, run_info = create_run(
        note="Run 4: val sweep after filtering invalid image paths",
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

    # Sweep thresholds
    thresholds = np.linspace(-0.5, 1.0, 200)

    print(f"Running threshold sweep on validation set (N={len(df)})...")
    sweep_df = sweep_thresholds(scores, labels, thresholds)

    # Save outputs
    out_dir = os.path.join("outputs", "sweeps")
    os.makedirs(out_dir, exist_ok=True)

    sweep_csv_path = os.path.join(out_dir, "val_sweep.csv")
    roc_path = os.path.join(out_dir, "roc_val.png")

    sweep_df.to_csv(sweep_csv_path, index=False)
    plot_roc(sweep_df, roc_path)

    best_idx = sweep_df["f1"].idxmax()
    best_row = sweep_df.loc[best_idx]
    log_artifact(run_dir, sweep_csv_path)
    log_artifact(run_dir, roc_path)

    append_run_csv(run_info)
    print(f"\nSaved sweep results to: {sweep_csv_path}")
    print(f"Saved ROC plot to: {roc_path}")
    print("\nBest threshold by max F1:")
    print(best_row[["threshold", "accuracy", "precision", "recall", "f1"]])


if __name__ == "__main__":
    main()