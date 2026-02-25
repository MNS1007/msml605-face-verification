import os
import json
import yaml
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


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

    print("Train shape:", train_df.shape)
    print("Validation shape:", val_df.shape)
    print("Test shape:", test_df.shape)

    return train_df, val_df, test_df


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

    print("Manifest written to ./dataset/dataset_manifest.json")


def main():
    config = load_config("config.yaml")

    file_path = "pairs.csv"

    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "jessicali9530/lfw-dataset",
        file_path
    )

    print("First 5 records:", df.head())

    train, val, test = split(df, config)

    write_manifest(file_path, df, train, val, test, config)


if __name__ == "__main__":
    main()
