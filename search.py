"""
search.py (FINAL)
--------------------------------------------------
✅ FAISS index preload (startup)
✅ SigLIP embedding 사용
✅ 그룹 평균 벡터 기반 검색
✅ cosine similarity 기준
✅ threshold 기반 매칭 판정
✅ Node 서버 연동용 안정 응답 구조
"""

import os
import numpy as np
import faiss
from PIL import Image

from siglip import image_to_vector


# ==================================================
# Path Config
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INDEX_PATH = os.path.join(BASE_DIR, "index", "siglip.index")
IDS_PATH = os.path.join(BASE_DIR, "index", "product_ids.npy")


# ==================================================
# Search Config (🔥 실서비스 기준)
# ==================================================

# cosine similarity 기준
SIMILARITY_THRESHOLD = 0.75   # ❗ 다시 올린다 (그룹 평균이라 안정)
TOP_K = 5


# ==================================================
# Load Assets (1회)
# ==================================================

def _load_assets():
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found: {INDEX_PATH}")

    if not os.path.exists(IDS_PATH):
        raise FileNotFoundError(f"product_ids not found: {IDS_PATH}")

    index = faiss.read_index(INDEX_PATH)
    product_ids = np.load(IDS_PATH, allow_pickle=True)

    if index.ntotal != len(product_ids):
        raise RuntimeError("Index size and product_ids length mismatch")

    return index, product_ids


# 🔥 서버 시작 시 1회 로드
INDEX, PRODUCT_IDS = _load_assets()


# ==================================================
# Search Logic
# ==================================================

def search_image(image_path: str, top_k: int = TOP_K):
    """
    업로드된 이미지 경로를 받아
    FAISS + SigLIP 기반으로 화장품 그룹 검색

    return 구조 (❗ Node 서버 의존):
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

    # ------------------------------
    # 1️⃣ 이미지 로드
    # ------------------------------
    img = Image.open(image_path).convert("RGB")

    # ------------------------------
    # 2️⃣ SigLIP embedding
    # ------------------------------
    q = image_to_vector(img).reshape(1, -1)  # (1, 512)

    # ------------------------------
    # 3️⃣ FAISS 검색 (cosine similarity)
    # ------------------------------
    sims, idxs = INDEX.search(q, top_k)

    results = []

    for sim, idx in zip(sims[0], idxs[0]):
        if idx < 0:
            continue

        pid = PRODUCT_IDS[int(idx)]
        similarity = float(sim)
        distance = float(1.0 - similarity)

        results.append({
            "product_id": str(pid),
            "similarity": similarity,
            "distance": distance,
        })

    # ------------------------------
    # 4️⃣ 결과 판정
    # ------------------------------
    if not results:
        return {
            "matched": False,
            "best": None,
            "results": [],
        }

    best = results[0]

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
