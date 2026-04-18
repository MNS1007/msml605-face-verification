import time
from src.embeddings import preprocess_image, get_embedding
from src.similarity import cosine_similarity


def compute_confidence(score: float, threshold: float) -> float:
    # simple, valid, explainable
    return abs(score - threshold)


def apply_threshold(score: float, threshold: float) -> bool:
    return score >= threshold


def infer_pair(img1_path, img2_path, threshold):
    start = time.time()

    # 1. preprocessing
    x1 = preprocess_image(img1_path)
    x2 = preprocess_image(img2_path)

    # 2. embedding
    e1 = get_embedding(x1)
    e2 = get_embedding(x2)

    # 3. similarity
    score = cosine_similarity(
        e1.reshape(1, -1),
        e2.reshape(1, -1)
    )[0]

    # 4. decision
    decision = apply_threshold(score, threshold)

    # 5. confidence
    confidence = compute_confidence(score, threshold)

    latency = time.time() - start

    return {
        "score": score,
        "decision": decision,
        "confidence": confidence,
        "latency": latency,
    }