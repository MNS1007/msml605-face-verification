"""Deterministic LFW ingestion: download, preprocess, split, and write manifest.

Usage:
    python scripts/ingest_lfw.py --config configs/m1.yaml
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def preprocess_pairs(df):
    """Clean messy CSV pair data into a uniform schema."""
    rows = []
    for _, row in df.iterrows():
        name = row["name"]
        img1 = row["imagenum1"]
        img2_raw = row["imagenum2"]
        col4 = row.get("Unnamed: 3", np.nan)

        try:
            img2 = int(float(img2_raw))
            rows.append({
                "name1": name,
                "imagenum1": int(img1),
                "name2": name,
                "imagenum2": img2,
                "label": 1,
            })
        except (ValueError, TypeError):
            rows.append({
                "name1": name,
                "imagenum1": int(img1),
                "name2": str(img2_raw),
                "imagenum2": int(float(col4)),
                "label": 0,
            })

    return pd.DataFrame(rows)


def split(df, config):
    """Deterministic train/val/test split using sklearn."""
    seed = config["seed"]
    ratios = config["split_ratios"]

    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - ratios["train"]),
        random_state=seed,
        shuffle=True,
    )

    val_size_adjusted = ratios["val"] / (ratios["val"] + ratios["test"])
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_size_adjusted),
        random_state=seed,
        shuffle=True,
    )
    return train_df, val_df, test_df


def write_manifest(output_dir, df, train, val, test, config, data_source):
    """Write dataset manifest with required fields."""
    manifest = {
        "data_source": data_source,
        "seed": config["seed"],
        "split_policy": config["split_policy"],
        "counts": {
            "total": len(df),
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "split_ratios": config["split_ratios"],
    }

    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Ingest LFW dataset")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = config.get("output_dir", "outputs")

    # Download dataset
    file_path = "pairs.csv"
    data_source = "jessicali9530/lfw-dataset"

    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        data_source,
        file_path,
    )
    data_path = kagglehub.dataset_download(data_source)

    # Save data_path to config so make_pairs.py can find images
    config["data_path"] = data_path
    with open(args.config, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Dataset downloaded to: {data_path}")
    print(df.head())

    # Preprocess and split
    df = preprocess_pairs(df)
    train, val, test = split(df, config)

    # Save split CSVs to outputs/
    os.makedirs(output_dir, exist_ok=True)
    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        split_df.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)
    print(f"Splits saved to {output_dir}/")

    # Write manifest
    write_manifest(output_dir, df, train, val, test, config, data_source)


if __name__ == "__main__":
    main()
