# siglip.py (EC2 최적화 최종본)
import torch
import numpy as np
from PIL import Image
import open_clip
import threading

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B-16-SigLIP-384"

_model = None
_preprocess = None
_lock = threading.Lock()


def _load_model():
    global _model, _preprocess
    print("🔥 Loading SigLIP model (one-time)...")

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained="webli"
    )
    model = model.to(DEVICE).eval()

    _model = model
    _preprocess = preprocess
    print("✅ SigLIP loaded")


def get_model():
    global _model, _preprocess
    if _model is None:
        with _lock:
            if _model is None:
                _load_model()
    return _model, _preprocess


def image_to_vector(img: Image.Image) -> np.ndarray:
    model, preprocess = get_model()

    img = img.convert("RGB")
    img_t = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feat = model.encode_image(img_t)
        feat = feat / feat.norm(dim=-1, keepdim=True)

    return feat.squeeze(0).cpu().numpy().astype("float32")
