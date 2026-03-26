import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.similarity import cosine_similarity


DEFAULT_CONFIG_PATH = os.path.join("configs", "m1.yaml")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)



def load_facenet_model(device: str):
    try:
        from facenet_pytorch import InceptionResnetV1
    except ImportError:
        raise ImportError(
            "facenet-pytorch is not installed.\n"
            "Run: pip install facenet-pytorch"
        )

    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    return model


def preprocess_image(img_path: str) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    img = img.resize((160, 160))

    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = (img_np - 0.5) / 0.5
    img_np = np.transpose(img_np, (2, 0, 1))

    return torch.tensor(img_np, dtype=torch.float32)


@torch.no_grad()
def compute_embeddings(model, img_paths, device: str, batch_size: int = 64):
    embeddings = []
    model.eval()

    for i in tqdm(range(0, len(img_paths), batch_size), desc="Embedding batches"):
        batch_paths = img_paths[i : i + batch_size]

        batch_imgs = []
        for p in batch_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Image path does not exist: {p}")
            batch_imgs.append(preprocess_image(p))

        batch_tensor = torch.stack(batch_imgs).to(device)
        batch_emb = model(batch_tensor)
        embeddings.append(batch_emb.cpu().numpy())

    return np.vstack(embeddings)


def validate_pairs_df(df: pd.DataFrame, split_name: str):
    required_cols = ["left_path", "right_path", "label", "split"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"[{split_name}] Missing required column: {col}")

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(f"[{split_name}] Invalid labels found: {df['label'].unique()}")

    if df.isnull().any().any():
        raise ValueError(f"[{split_name}] Found NaN values in pairs file")


def score_split(model, device: str, pairs_csv_path: str, output_csv_path: str, batch_size: int = 64):
    split_name = os.path.basename(pairs_csv_path).replace(".csv", "")

    print(f"\nLoading pairs: {pairs_csv_path}")
    df = pd.read_csv(pairs_csv_path)
    validate_pairs_df(df, split_name)

    left_paths = df["left_path"].tolist()
    right_paths = df["right_path"].tolist()

    print(f"Computing embeddings for {split_name} (N={len(df)}) ...")
    left_emb = compute_embeddings(model, left_paths, device=device, batch_size=batch_size)
    right_emb = compute_embeddings(model, right_paths, device=device, batch_size=batch_size)

    print("Computing cosine similarity scores...")
    scores = cosine_similarity(left_emb, right_emb)

    if len(scores) != len(df):
        raise RuntimeError(f"[{split_name}] Score count mismatch")

    df_out = df.copy()
    df_out["score"] = scores

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_out.to_csv(output_csv_path, index=False)

    print(f"Saved scored split to: {output_csv_path}")
    print(f"[{split_name}] score stats: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")


def main():
    cfg = load_config(DEFAULT_CONFIG_PATH)
    output_dir = cfg.get("output_dir", "outputs")

    pairs_dir = os.path.join(output_dir, "pairs")
    scores_dir = os.path.join(output_dir, "scores")

    train_pairs = os.path.join(pairs_dir, "train.csv")
    val_pairs = os.path.join(pairs_dir, "val.csv")
    test_pairs = os.path.join(pairs_dir, "test.csv")

    for p in [train_pairs, val_pairs, test_pairs]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Pairs file missing: {p}. Run make_pairs.py first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading FaceNet model...")
    model = load_facenet_model(device=device)

    score_split(
        model=model,
        device=device,
        pairs_csv_path=train_pairs,
        output_csv_path=os.path.join(scores_dir, "train_scored.csv"),
    )

    score_split(
        model=model,
        device=device,
        pairs_csv_path=val_pairs,
        output_csv_path=os.path.join(scores_dir, "val_scored.csv"),
    )

    score_split(
        model=model,
        device=device,
        pairs_csv_path=test_pairs,
        output_csv_path=os.path.join(scores_dir, "test_scored.csv"),
    )

    print("\nDone. All splits scored.")


if __name__ == "__main__":
    main()