import os
import json
import yaml
import numpy as np
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split

# Load the config
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# I added this function because the dataset was getting messy (strings in imagenum2 column)
def preprocess_pairs(df):
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
                "label": 1
            })
        except (ValueError, TypeError):
            rows.append({
                "name1": name,
                "imagenum1": int(img1),
                "name2": str(img2_raw),
                "imagenum2": int(float(col4)),
                "label": 0
            })

    clean_df = pd.DataFrame(rows)
    return clean_df

# Pulls the seed and split ratios from the config and performs the split
def split(df, config):
    seed = config["seed"]
    split_ratios = config["split_ratios"]

    train_ratio = split_ratios["train"]
    val_ratio = split_ratios["val"]

    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - train_ratio),
        random_state=seed,
        shuffle=True
    )

    val_size_adjusted = val_ratio / (val_ratio + split_ratios["test"])

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_size_adjusted),
        random_state=seed,
        shuffle=True
    )
    return train_df, val_df, test_df

# Gets the dataframes and writes the stats to a manifest
def write_manifest(filepath, df, train, val, test, config):
    manifest = {
        "dataset_name": f"jessicali9530/lfw-dataset/{filepath}",
        "total_samples": len(df),
        "train_samples": len(train),
        "val_samples": len(val),
        "test_samples": len(test),
        "split_policy": config["split_policy"],
        "split_ratios": config["split_ratios"],
        "random_seed": config["seed"],
        "shuffle": True
    }

    os.makedirs("./dataset", exist_ok=True)

    with open("./dataset/dataset_manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)
    print("Manifest saved to ./dataset/dataset_manifest.json")

def main():
    config = load_config("./configs/m1.yaml")

    file_path = "pairs.csv"
    output_dir="./dataset"

    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "jessicali9530/lfw-dataset",
        file_path
    )
    path = kagglehub.dataset_download("jessicali9530/lfw-dataset")
     # Downloads the images to cache, then writes it to config so that it can be used in make_pairs.py
    config["data_path"] = path
    with open("C:/Users/srira/OneDrive/Desktop/604/milestone-1/configs/m1.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(df.head())

    df = preprocess_pairs(df)

    train, val, test = split(df, config)

    os.makedirs(output_dir, exist_ok=True)
    # Saves the splits to the output directory
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)
    print("Splits saved to ./dataset")
    write_manifest(file_path, df, train, val, test, config)


if __name__ == "__main__":
    main()
