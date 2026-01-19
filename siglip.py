"""
siglip.py (FINAL STABLE)
--------------------------------------------------
✅ SigLIP 모델 startup preload
✅ GPU / CPU 자동 선택
✅ torch.no_grad + eval 고정
✅ float32 고정 출력
✅ cosine similarity 전제 (L2 normalize)
✅ search.py 최종본과 100% 호환
"""

import torch
import numpy as np
from PIL import Image
import open_clip

# ==================================================
# Device / Model Config
# ==================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B-16-SigLIP-384"

_model = None
_preprocess = None


# ==================================================
# Model Loader (startup 1회)
# ==================================================
def load_model():
    """
    - FastAPI startup 시 1회만 호출
    - search.py 에서 직접 호출하지 않음
    """
    global _model, _preprocess

    if _model is not None:
        return

    print("🔥 Loading SigLIP model (startup preload)...")

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained="webli"
    )

    model = model.to(DEVICE)
    model.eval()  # 🔥 inference 고정

    _model = model
    _preprocess = preprocess

    print("✅ SigLIP loaded (device=%s)" % DEVICE)


# ==================================================
# Image → Embedding
# ==================================================
def image_to_vector(img: Image.Image) -> np.ndarray:
    """
    입력:
        PIL.Image (RGB 권장)
    출력:
        np.ndarray (float32, L2-normalized, shape=(D,))
    """

    if _model is None or _preprocess is None:
        raise RuntimeError("SigLIP model not loaded. Call load_model() first.")

    # PIL 안정화
    if img.mode != "RGB":
        img = img.convert("RGB")

    # preprocess → tensor
    img_t = _preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feat = _model.encode_image(img_t)
        feat = feat / feat.norm(dim=-1, keepdim=True)  # 🔥 cosine 전제

    # numpy 반환 (FAISS / numpy 연산용)
    return (
        feat
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
        .astype("float32")
    )
