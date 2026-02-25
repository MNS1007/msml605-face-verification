import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split
import json

def split(df, seed):
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=seed,
        shuffle=True
    )

    print("Train shape:", train_df.shape)
    print("Validation shape:", val_df.shape)

    return train_df, val_df

def manifest(filepath, df, train, val, seed):
    manifest = {
        "dataset_name": "jessicali9530/lfw-dataset/{filepath}",
        "total_samples": len(df),
        "train_samples": len(train),
        "val_samples": len(val),
        "split_policy": "random split",
        "test_size": 0.2,
        "random_seed": seed,
        "shuffle": True
    }

    with open("./dataset/dataset_manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)

def main():
    file_path = "pairs.csv"
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "jessicali9530/lfw-dataset",
        file_path
    )
    print("First 5 records:", df.head())

    seed = 100

    train, val = split(df, seed)
    manifest(file_path, df, train, val, seed)
    

if __name__ == "__main__":
    main()
