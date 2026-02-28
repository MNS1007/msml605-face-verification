import os
import yaml
import pandas as pd

# Loads the config
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Makes the filepath for each persons image from your local cache
def resolve_image_path(data_path, name, imagenum):
    filename = f"{name}_{int(imagenum):04d}.jpg"
    return os.path.join(data_path, "lfw-deepfunneled", "lfw-deepfunneled", name, filename)


def process_split(split_csv_path, split_name, data_path):
    df = pd.read_csv(split_csv_path)
    # Use lambda function to apply the image path function to each name in each row
    df["left_path"] = df.apply(
        lambda row: resolve_image_path(data_path, row["name1"], row["imagenum1"]),
        axis=1
    )
    df["right_path"] = df.apply(
        lambda row: resolve_image_path(data_path, row["name2"], row["imagenum2"]),
        axis=1
    )
    df = df.sort_values(by=["left_path", "right_path"]).reset_index(drop=True)
    pairs_df = df[["left_path", "right_path", "label"]]

    return pairs_df




def main():
    config = load_config("./configs/m1.yaml")
    data_path = config["data_path"]
    seed = config["seed"]

    dataset_dir = "./dataset"
    output_dir = "./dataset/pairs" # Creates a new folder inside dataset for the pairs
    os.makedirs(output_dir, exist_ok=True)

    splits = {
        "train": os.path.join(dataset_dir, "train.csv"),
        "val": os.path.join(dataset_dir, "val.csv"),
        "test": os.path.join(dataset_dir, "test.csv"),
    }
    # Creates the pairs for each split
    for split_name, split_csv_path in splits.items():
        pairs_df = process_split(split_csv_path, split_name, data_path)
        output_path = os.path.join(output_dir, f"{split_name}_pairs.csv")
        pairs_df.to_csv(output_path, index=False)

        pos_count = (pairs_df["label"] == 1).sum()
        neg_count = (pairs_df["label"] == 0).sum()



if __name__ == "__main__":
    main()
