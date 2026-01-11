"""
siglip.py (EC2 배포 안정 최종본)
--------------------------------------------------
✅ SigLIP(OpenCLIP) 모델 로드
✅ image_to_vector(img) -> float32 numpy 벡터 반환
✅ CPU/ CUDA 자동 선택
"""

import torch
import numpy as np
from PIL import Image
import open_clip

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B-16-SigLIP-384"
PRETRAINED = "webli"

# ✅ 모델은 프로세스 시작 시 1회 로드
model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME,
    pretrained=PRETRAINED
)
model = model.to(DEVICE).eval()


def image_to_vector(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    img_t = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feat = model.encode_image(img_t)
        feat = feat / feat.norm(dim=-1, keepdim=True)

    return feat.squeeze(0).cpu().numpy().astype("float32")
