"""Apply data-centric improvements to verification pairs.

Usage:
    python scripts/apply_data_centric.py --config configs/m2.yaml

Reads the original pairs from outputs/pairs/, applies identity capping,
missing-image filtering, and label rebalancing, then writes improved
pairs to outputs/pairs_improved/.
"""

import argparse
import json
import os

import pandas as pd
import yaml

from src.data_centric import apply_all_improvements, count_identity_appearances


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Apply data-centric improvements")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = config.get("output_dir", "outputs")
    pairs_dir = os.path.join(output_dir, "pairs")
    improved_dir = os.path.join(output_dir, "pairs_improved")
    os.makedirs(improved_dir, exist_ok=True)

    seed = config.get("seed", 42)
    dc_config = config.get("data_centric", {})
    max_pairs = dc_config.get("max_pairs_per_identity", 30)
    check_images = dc_config.get("check_images", True)

    summary = {}

    for split_name in ["train", "val", "test"]:
        input_path = os.path.join(pairs_dir, f"{split_name}.csv")
        if not os.path.isfile(input_path):
            print(f"Skipping {split_name}: {input_path} not found")
            continue

        print(f"\n--- Processing {split_name} split ---")
        df = pd.read_csv(input_path)
        original_count = len(df)

        # Show top identities before improvement
        id_counts = count_identity_appearances(df)
        top5 = id_counts.head(5)
        print(f"Top 5 identities before: {dict(top5)}")

        # Apply improvements
        improved = apply_all_improvements(
            df,
            max_pairs_per_identity=max_pairs,
            seed=seed,
            check_images=check_images,
        )

        # Show top identities after improvement
        id_counts_after = count_identity_appearances(improved)
        top5_after = id_counts_after.head(5)
        print(f"Top 5 identities after:  {dict(top5_after)}")

        # Save improved pairs
        output_path = os.path.join(improved_dir, f"{split_name}.csv")
        improved.to_csv(output_path, index=False)
        print(f"Saved {len(improved)} pairs to {output_path}")

        summary[split_name] = {
            "original_pairs": original_count,
            "improved_pairs": len(improved),
            "removed": original_count - len(improved),
            "positive": int((improved["label"] == 1).sum()),
            "negative": int((improved["label"] == 0).sum()),
        }

    # Save summary
    summary_path = os.path.join(improved_dir, "data_centric_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "description": (
                    "Data-centric improvement: capped overrepresented identities "
                    f"at {max_pairs} pairs per identity, filtered missing images, "
                    "rebalanced positive/negative labels to 1:1 ratio."
                ),
                "max_pairs_per_identity": max_pairs,
                "seed": seed,
                "check_images": check_images,
                "splits": summary,
            },
            f,
            indent=2,
        )
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
