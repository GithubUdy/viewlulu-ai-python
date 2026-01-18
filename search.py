"""
search.py (FINAL)
--------------------------------------------------
✅ FAISS index preload
✅ SigLIP embedding 사용
✅ similarity / distance 명확 분리
✅ match 판단을 Python에서 수행
✅ EC2 / 실행 위치 무관 경로 안정
"""

import os
import faiss
import numpy as np
from PIL import Image

from siglip import image_to_vector

# ===============================
# Path Config
# ===============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INDEX_PATH = os.path.join(BASE_DIR, "index", "siglip.index")
IDS_PATH = os.path.join(BASE_DIR, "index", "product_ids.npy")

# ===============================
# Search Config
# ===============================

SIMILARITY_THRESHOLD = 0.75   # 🔥 실서비스 기준 안정값 (0.7 ~ 0.8)

# ===============================
# Load Assets (1회)
# ===============================

def _load_assets():
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found: {INDEX_PATH}")

    if not os.path.exists(IDS_PATH):
        raise FileNotFoundError(f"product_ids not found: {IDS_PATH}")

    index = faiss.read_index(INDEX_PATH)
    product_ids = np.load(IDS_PATH, allow_pickle=True)

    return index, product_ids


# 🔥 서버 시작 시 1회 로드
INDEX, PRODUCT_IDS = _load_assets()

# ===============================
# Search Logic
# ===============================

def search_image(image_path: str, top_k: int = 5):
    """
    업로드된 이미지 경로를 받아
    FAISS + SigLIP 기반으로 유사 화장품 검색

    return:
    {
        "matched": bool,
        "best": {
            "product_id": str,
            "similarity": float,
            "distance": float
        } | None,
        "results": [
            {
                "product_id": str,
                "similarity": float,
                "distance": float
            },
            ...
        ]
    }
    """

    if top_k <= 0:
        top_k = 5

    # 1️⃣ 이미지 로드
    img = Image.open(image_path).convert("RGB")

    # 2️⃣ SigLIP embedding
    q = image_to_vector(img).reshape(1, -1)

    # 3️⃣ FAISS 검색 (cosine similarity)
    sims, idxs = INDEX.search(q, top_k)

    results = []
    for sim, idx in zip(sims[0], idxs[0]):
        if int(idx) < 0:
            continue

        pid = PRODUCT_IDS[int(idx)]

        results.append({
            "product_id": str(pid),
            "similarity": float(sim),
            "distance": float(1.0 - sim),
        })

    # 4️⃣ 결과 판단
    best = results[0] if results else None

    if not best:
        return {
            "matched": False,
            "best": None,
            "results": results,
        }

    if best["similarity"] < SIMILARITY_THRESHOLD:
        return {
            "matched": False,
            "best": best,
            "results": results,
        }

    return {
        "matched": True,
        "best": best,
        "results": results,
    }
