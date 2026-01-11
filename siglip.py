# siglip.py
import torch
import numpy as np
from PIL import Image
import open_clip

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B-16-SigLIP-384"

_model = None
_preprocess = None


def load_model():
    global _model, _preprocess
    if _model is not None:
        return

    print("🔥 Loading SigLIP model (startup preload)...")

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained="webli"
    )
    model = model.to(DEVICE).eval()

    _model = model
    _preprocess = preprocess

    print("✅ SigLIP loaded")


def image_to_vector(img: Image.Image) -> np.ndarray:
    if _model is None:
        raise RuntimeError("SigLIP model not loaded")

    img = img.convert("RGB")
    img_t = _preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feat = _model.encode_image(img_t)
        feat = feat / feat.norm(dim=-1, keepdim=True)

    return feat.squeeze(0).cpu().numpy().astype("float32")
