"""
siglip.py (EC2 안정화 최종본)
--------------------------------------------------
✅ 서버 시작 시 모델 1회 로딩
✅ 요청 시 재로딩 없음
✅ CPU 강제 사용
✅ 메모리 사용 최소화
"""

import torch
import numpy as np
from PIL import Image
import open_clip

# ==============================
# 환경 고정
# ==============================
DEVICE = "cpu"  # EC2에서는 무조건 CPU
MODEL_NAME = "ViT-B-16-SigLIP-384"

# ==============================
# 모델 1회 로딩 (서버 시작 시)
# ==============================
print("🔥 Loading SigLIP model (one-time)...")

model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME,
    pretrained="webli"
)

model = model.to(DEVICE)
model.eval()

print("✅ SigLIP model loaded and ready")

# ==============================
# 이미지 → 벡터 변환
# ==============================
def image_to_vector(img: Image.Image) -> np.ndarray:
    """
    PIL Image → normalized embedding vector (float32)
    """
    img = img.convert("RGB")
    img_t = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feat = model.encode_image(img_t)
        feat = feat / feat.norm(dim=-1, keepdim=True)

    return feat.squeeze(0).cpu().numpy().astype("float32")
