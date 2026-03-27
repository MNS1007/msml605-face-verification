"""Run error analysis on scored verification pairs.

Usage:
    python scripts/run_error_analysis.py

Reads the scored validation or test pairs, applies the selected threshold,
and produces error slice analysis with two defined slices:
  Slice 1: False negatives among same-identity pairs with rare identities
  Slice 2: False positives near the decision boundary

Outputs JSON report and prints summary to stdout.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.error_analysis import run_error_analysis
from src.tracking.run_logger import create_run, log_metrics, append_run_csv


def main():
    # Load threshold
    threshold_path = os.path.join("outputs", "threshold", "selected_threshold.json")
    if not os.path.exists(threshold_path):
        raise FileNotFoundError(f"Missing: {threshold_path}. Run select_threshold.py first.")

    with open(threshold_path) as f:
        threshold_data = json.load(f)
    threshold = threshold_data["threshold"]

    # Load scored validation data for error analysis
    val_scored_path = os.path.join("outputs", "scores", "val_scored.csv")
    if not os.path.exists(val_scored_path):
        raise FileNotFoundError(f"Missing: {val_scored_path}. Run score_pairs.py first.")

    df = pd.read_csv(val_scored_path)
    scores = df["score"].to_numpy()
    labels = df["label"].to_numpy().astype(int)
    predictions = (scores >= threshold).astype(int)

    # Run error analysis
    print(f"Running error analysis on validation set (N={len(df)}, threshold={threshold:.4f})...")
    analysis = run_error_analysis(
        df=df,
        scores=scores,
        predictions=predictions,
        threshold=threshold,
        max_images=4,
        boundary_width=0.05,
    )

    # Print results
    for s in analysis["slices"]:
        print(f"\n{'='*60}")
        print(f"SLICE: {s['slice_name']}")
        print(f"Definition: {s['definition']}")
        print(f"Total in slice: {s['total_in_slice']}")
        print(f"Errors in slice: {s['errors_in_slice']}")
        print(f"Error rate: {s['error_rate']:.4f}")
        print(f"Hypothesis: {s['hypothesis']}")
        print(f"Future improvement: {s['future_improvement']}")
        if s["examples"]:
            print(f"\nRepresentative examples ({len(s['examples'])}):")
            for ex in s["examples"]:
                print(f"  {ex}")

    # Save output
    out_dir = os.path.join("outputs", "error_analysis")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "error_analysis.json")
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved error analysis to: {out_path}")

    # Log as tracked run
    run_id, run_dir, run_info = create_run(
        note="Error analysis: 2 slices on validation set",
        data_version="filtered_pairs_v1",
    )
    with open(os.path.join(run_dir, "error_analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2)
    append_run_csv(run_info)


if __name__ == "__main__":
    main()
