"""Generate verification pairs from dataset splits.

Usage:
    python scripts/make_pairs.py --config configs/m1.yaml
"""

import argparse
import os

import pandas as pd
import yaml


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_image_path(data_path, name, imagenum):
    """Build full path to an LFW image from the local cache."""
    filename = f"{name}_{int(imagenum):04d}.jpg"
    return os.path.join(data_path, "lfw-deepfunneled", "lfw-deepfunneled", name, filename)

# Updated the process function for the data change
def process_split(split_csv_path, split_name, data_path):
    df = pd.read_csv(split_csv_path)

    df["left_path"] = df.apply(
        lambda row: resolve_image_path(data_path, row["name1"], row["imagenum1"]),
        axis=1,
    )
    df["right_path"] = df.apply(
        lambda row: resolve_image_path(data_path, row["name2"], row["imagenum2"]),
        axis=1,
    )
    df["split"] = split_name
    df = df.sort_values(by=["left_path", "right_path"]).reset_index(drop=True)

    before = len(df)

    df = df[
        df["left_path"].apply(os.path.exists) &
        df["right_path"].apply(os.path.exists)
    ]

    after = len(df)

    print(f"[{split_name}] Removed {before - after} invalid pairs ({before} -> {after})")

    return df[["left_path", "right_path", "label", "split"]]

def main():
    parser = argparse.ArgumentParser(description="Generate verification pairs")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    data_path = config["data_path"]
    output_dir = config.get("output_dir", "outputs")
    pairs_dir = os.path.join(output_dir, "pairs")
    os.makedirs(pairs_dir, exist_ok=True)

    splits = {
        "train": os.path.join(output_dir, "train.csv"),
        "val": os.path.join(output_dir, "val.csv"),
        "test": os.path.join(output_dir, "test.csv"),
    }

    for split_name, split_csv_path in splits.items():
        pairs_df = process_split(split_csv_path, split_name, data_path)
        output_path = os.path.join(pairs_dir, f"{split_name}.csv")
        pairs_df.to_csv(output_path, index=False)

        pos = (pairs_df["label"] == 1).sum()
        neg = (pairs_df["label"] == 0).sum()
        print(f"{split_name}: {len(pairs_df)} pairs ({pos} positive, {neg} negative) -> {output_path}")


if __name__ == "__main__":
    main()
