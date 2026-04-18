"""Pair-level inference with separated stages.

Stages: preprocessing -> embedding -> similarity -> threshold -> confidence.
Each stage is visible and independently testable.
"""

import time

from src.embeddings import preprocess_image, get_embedding
from src.similarity import cosine_similarity
from src.confidence import calibrated_confidence


def apply_threshold(score: float, threshold: float) -> bool:
    """Return True (same) when score >= threshold."""
    return score >= threshold


def infer_pair(img1_path, img2_path, threshold):
    """Run full inference on one image pair.

    Returns dict with score, decision, confidence (calibrated, 0.5-1.0),
    and latency in seconds.
    """
    start = time.time()

    # 1. preprocessing
    x1 = preprocess_image(img1_path)
    x2 = preprocess_image(img2_path)

    # 2. embedding
    e1 = get_embedding(x1)
    e2 = get_embedding(x2)

    # 3. similarity
    score = float(cosine_similarity(
        e1.reshape(1, -1),
        e2.reshape(1, -1)
    )[0])

    # 4. decision
    decision = apply_threshold(score, threshold)

    # 5. calibrated confidence (see src/confidence.py for formula)
    confidence = calibrated_confidence(score, threshold)

    latency = time.time() - start

    return {
        "score": score,
        "decision": decision,
        "confidence": confidence,
        "latency": latency,
    }
