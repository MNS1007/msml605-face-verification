import numpy as np
import pytest
import pandas as pd
from src.embeddings import preprocess_image, get_embedding
from src.similarity import cosine_similarity
from src.inference import apply_threshold
from src.confidence import calibrated_confidence
import os

# -------------------------------------------------------------------------
# Helpers: get two real images from your existing dataset/pairs
# -------------------------------------------------------------------------
def get_two_images():
    pairs_path = "outputs/pairs/test.csv"

    assert os.path.exists(pairs_path), (
        "Pairs file not found. Run make_pairs.py first."
    )

    df = pd.read_csv(pairs_path)

    # Find first valid pair (paths exist)
    for _, row in df.iterrows():
        if os.path.exists(row["left_path"]) and os.path.exists(row["right_path"]):
            return row["left_path"], row["right_path"]

    raise RuntimeError("No valid image paths found in pairs file.")
# -------------------------------------------------------------------------
# Embedding test (real pipeline)
# -------------------------------------------------------------------------
def test_embedding_output_shape():
    img1, _ = get_two_images()

    x = preprocess_image(img1)
    emb = get_embedding(x)

    assert emb.ndim == 1
    assert emb.shape[0] > 0


# -------------------------------------------------------------------------
# Similarity test (embedding-based)
# -------------------------------------------------------------------------
def test_cosine_similarity_valid_range():
    img1, img2 = get_two_images()

    e1 = get_embedding(preprocess_image(img1))
    e2 = get_embedding(preprocess_image(img2))

    sim = cosine_similarity(
        e1.reshape(1, -1),
        e2.reshape(1, -1)
    )[0]

    assert -1.0 <= sim <= 1.0


# -------------------------------------------------------------------------
# Threshold logic test
# -------------------------------------------------------------------------
def test_threshold_application():
    assert apply_threshold(0.7, 0.5) is True
    assert apply_threshold(0.3, 0.5) is False


# -------------------------------------------------------------------------
# Confidence calculation test
# -------------------------------------------------------------------------
def test_confidence_computation():
    conf = calibrated_confidence(0.7, 0.5)
    # Score 0.7, threshold 0.5: above threshold, confidence > 0.5
    assert 0.5 < conf <= 1.0


# -------------------------------------------------------------------------
# End-to-end consistency check (lightweight)
# -------------------------------------------------------------------------
def test_embedding_similarity_consistency():
    img1, img2 = get_two_images()

    e1 = get_embedding(preprocess_image(img1))
    e2 = get_embedding(preprocess_image(img2))

    sim1 = cosine_similarity(e1.reshape(1, -1), e2.reshape(1, -1))[0]
    sim2 = cosine_similarity(e1.reshape(1, -1), e2.reshape(1, -1))[0]

    # deterministic check
    assert sim1 == pytest.approx(sim2)