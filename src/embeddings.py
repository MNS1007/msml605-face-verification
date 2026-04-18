import torch
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1

_model = InceptionResnetV1(pretrained='vggface2').eval()


def preprocess_image(image_path: str) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((160, 160))

    x = np.array(img) / 255.0
    x = torch.tensor(x).permute(2, 0, 1).float().unsqueeze(0)

    return x


def get_embedding(image_tensor: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        emb = _model(image_tensor)
    return emb.squeeze().numpy()


def embed_image(image_path: str) -> np.ndarray:
    x = preprocess_image(image_path)
    return get_embedding(x)