import torch
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1

_model = InceptionResnetV1(pretrained='vggface2').eval()


def preprocess_image(image_path: str) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((160, 160))

    # FaceNet (VGGFace2) expects inputs normalized to [-1, 1].
    # This matches scripts/score_pairs.py so the calibrated threshold
    # selected against scored CSVs applies to the inference path.
    x = np.array(img).astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    x = np.transpose(x, (2, 0, 1))
    return torch.tensor(x, dtype=torch.float32).unsqueeze(0)


def get_embedding(image_tensor: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        emb = _model(image_tensor)
    return emb.squeeze().numpy()


def embed_image(image_path: str) -> np.ndarray:
    x = preprocess_image(image_path)
    return get_embedding(x)