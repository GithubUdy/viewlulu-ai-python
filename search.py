"""
search.py (FINAL)
--------------------------------------------------
✅ FAISS index preload (기존 유지)
✅ SigLIP embedding 사용
✅ 전체 DB 검색 + 그룹 단위 검색 공존
✅ 그룹 비교는 "상위 2개 평균" 방식
"""

import os
import faiss
import numpy as np
from PIL import Image
from typing import Dict, List

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

SIMILARITY_THRESHOLD = 0.55  # 기존 유지

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


INDEX, PRODUCT_IDS = _load_assets()

# ===============================
# Utility
# ===============================

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b))


def avg_of_best_two(distances: List[float]) -> float:
    if not distances:
        return float("inf")
    if len(distances) == 1:
        return distances[0]

    sorted_d = sorted(distances)
    return (sorted_d[0] + sorted_d[1]) / 2


# ===============================
# 기존: 전체 DB 검색 (유지)
# ===============================

def search_image(image_path: str, top_k: int = 5):
    if top_k <= 0:
        top_k = 5

    img = Image.open(image_path).convert("RGB")
    q = image_to_vector(img).reshape(1, -1)

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

    best = results[0] if results else None

    if not best or best["similarity"] < SIMILARITY_THRESHOLD:
        return {"matched": False, "best": best, "results": results}

    return {"matched": True, "best": best, "results": results}


# ===============================
# 🔥 신규: 그룹 단위 비교 (파우치 전용)
# ===============================

def search_image_with_groups(
    image_path: str,
    groups: Dict[str, List[str]],
):
    """
    groups = {
        "12": ["img1.jpg", "img2.jpg"],
        "15": ["img1.jpg", "img2.jpg", "img3.jpg"]
    }
    """

    img = Image.open(image_path).convert("RGB")
    query_vec = image_to_vector(img)

    best_group_id = None
    best_score = float("inf")

    for group_id, image_paths in groups.items():
        distances: List[float] = []

        for path in image_paths:
            try:
                g_img = Image.open(path).convert("RGB")
                g_vec = image_to_vector(g_img)
                dist = cosine_distance(query_vec, g_vec)
                distances.append(dist)
            except Exception:
                continue

        if not distances:
            continue

        group_score = avg_of_best_two(distances)

        if group_score < best_score:
            best_score = group_score
            best_group_id = group_id

    return {
        "matched": best_group_id is not None,
        "group_id": best_group_id,
        "score": best_score,
    }
