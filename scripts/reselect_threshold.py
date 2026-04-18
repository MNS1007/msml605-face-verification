#!/usr/bin/env python3
"""Re-select the operating threshold using the Milestone 3 embedding pipeline.

Reads validation pairs, computes embeddings via src/embeddings.py,
scores via cosine similarity, sweeps thresholds, and selects by max F1.
Saves updated threshold JSON and sweep CSV.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.embeddings import embed_image
from src.similarity import cosine_similarity
from src.evaluation import threshold_sweep, select_threshold


def main():
    config_path = os.path.join("configs", "m3.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg.get("output_dir", "outputs")
    val_pairs_path = os.path.join(output_dir, "pairs", "val.csv")

    if not os.path.exists(val_pairs_path):
        sys.exit(f"Validation pairs not found: {val_pairs_path}")

    df = pd.read_csv(val_pairs_path)
    print(f"Loaded {len(df)} validation pairs from {val_pairs_path}")

    # Compute embeddings for all unique images
    all_paths = list(set(df["left_path"].tolist() + df["right_path"].tolist()))
    print(f"Computing embeddings for {len(all_paths)} unique images...")
    emb_cache = {}
    for p in tqdm(all_paths, desc="Embedding"):
        if os.path.exists(p):
            emb_cache[p] = embed_image(p)

    # Score each pair
    scores = []
    valid_mask = []
    for _, row in df.iterrows():
        lp, rp = row["left_path"], row["right_path"]
        if lp in emb_cache and rp in emb_cache:
            s = cosine_similarity(
                emb_cache[lp].reshape(1, -1),
                emb_cache[rp].reshape(1, -1),
            )[0]
            scores.append(float(s))
            valid_mask.append(True)
        else:
            scores.append(0.0)
            valid_mask.append(False)

    df["score"] = scores
    df["valid"] = valid_mask
    df_valid = df[df["valid"]].copy()
    print(f"Valid scored pairs: {len(df_valid)} / {len(df)}")

    labels = df_valid["label"].values
    scores_arr = df_valid["score"].values

    # Sweep thresholds
    tcfg = cfg.get("threshold", {})
    t_min = tcfg.get("min", -1.0)
    t_max = tcfg.get("max", 1.0)
    t_steps = tcfg.get("steps", 201)
    rule = tcfg.get("selection_rule", "max_f1")
    higher_is_same = tcfg.get("higher_is_same", True)

    thresholds = np.linspace(t_min, t_max, t_steps)
    print(f"Sweeping {t_steps} thresholds [{t_min}, {t_max}], rule={rule}")

    sweep = threshold_sweep(labels, scores_arr, thresholds, higher_is_same=higher_is_same)
    best = select_threshold(sweep, rule=rule)

    print(f"\nSelected threshold: {best['threshold']:.6f}")
    print(f"  F1:       {best['f1']:.4f}")
    print(f"  Accuracy: {best['accuracy']:.4f}")
    print(f"  Precision:{best['precision']:.4f}")
    print(f"  Recall:   {best['recall']:.4f}")

    # Save results
    thresh_dir = os.path.join(output_dir, "threshold")
    os.makedirs(thresh_dir, exist_ok=True)

    result = {
        "selection_rule": f"{rule}_on_validation",
        "threshold": best["threshold"],
        "embedding_pipeline": "src/embeddings.py (FaceNet VGGFace2)",
        "preprocessing": "[0, 1] normalized, 160x160 resize",
        "metrics": {
            "accuracy": best["accuracy"],
            "precision": best["precision"],
            "recall": best["recall"],
            "f1": best["f1"],
        },
    }
    out_path = os.path.join(thresh_dir, "selected_threshold.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_path}")

    # Save sweep CSV
    sweep_dir = os.path.join(output_dir, "sweeps")
    os.makedirs(sweep_dir, exist_ok=True)
    sweep_df = pd.DataFrame(sweep)
    sweep_path = os.path.join(sweep_dir, "m3_val_sweep.csv")
    sweep_df.to_csv(sweep_path, index=False)
    print(f"Saved: {sweep_path}")

    # Update m3.yaml inference threshold
    cfg["inference"]["threshold"] = round(best["threshold"], 4)
    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Updated {config_path} inference.threshold = {best['threshold']:.4f}")


if __name__ == "__main__":
    main()
