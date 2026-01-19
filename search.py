"""
search.py (FINAL FAST + AUTO TUNED)
--------------------------------------------------
✅ FAISS index preload (startup)
✅ SigLIP embedding (query only, 1회)
✅ FAISS 후보 그룹 축소 (TOP_K)
✅ 그룹별 1:N(max) 비교
✅ 자동 threshold 튜닝 (min + gap)
✅ Node 연동 안정 응답
"""

import os
import logging
import numpy as np
import faiss
from PIL import Image
from siglip import image_to_vector


# ==================================================
# Logger
# ==================================================
logger = logging.getLogger(__name__)


# ==================================================
# Path Config
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INDEX_PATH = os.path.join(BASE_DIR, "index", "siglip.index")
IDS_PATH = os.path.join(BASE_DIR, "index", "product_ids.npy")


# ==================================================
# Search Config
# ==================================================
TOP_K = 20              # FAISS 후보 개수
MIN_THRESHOLD = 0.45    # 절대 최소 점수
GAP_THRESHOLD = 0.07    # 1등-2등 점수 차이


# ==================================================
# Load Assets (1회)
# ==================================================
def _load_assets():
    index = faiss.read_index(INDEX_PATH)
    product_ids = np.load(IDS_PATH, allow_pickle=True)

    if index.ntotal != len(product_ids):
        raise RuntimeError("Index size mismatch")

    logger.info("[FAISS] index loaded total=%d", index.ntotal)
    return index, product_ids


INDEX, PRODUCT_IDS = _load_assets()


# ==================================================
# 전체 DB 검색 (기존 유지 ❌ 변경 금지)
# ==================================================

def search_image(image_path: str, top_k: int = TOP_K):
    img = Image.open(image_path).convert("RGB")
    q = image_to_vector(img).reshape(1, -1)

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

# ==================================================
# 🔥 사용자 파우치 그룹 검색 (FAST)
# ==================================================
def search_image_with_groups(image_path: str, groups: dict):
    """
    image_path: 촬영 이미지
    groups: {
        "12": ["/tmp/12/1.jpg", "/tmp/12/2.jpg", ...],
        ...
    }
    """

    logger.info(
        "[GROUP_SEARCH][START] groups=%d image=%s",
        len(groups),
        os.path.basename(image_path),
    )

    # --------------------------------------------------
    # 1️⃣ Query embedding (1회)
    # --------------------------------------------------
    img = Image.open(image_path).convert("RGB")
    q = image_to_vector(img).astype("float32")
    q /= np.linalg.norm(q)
    q = q.reshape(1, -1)

    # --------------------------------------------------
    # 2️⃣ FAISS 후보 축소
    # --------------------------------------------------
    sims, idxs = INDEX.search(q, TOP_K)

    candidate_groups = {}
    for sim, idx in zip(sims[0], idxs[0]):
        if idx < 0:
            continue

        group_id = str(PRODUCT_IDS[int(idx)])
        candidate_groups.setdefault(group_id, []).append(float(sim))

    logger.info(
        "[GROUP_SEARCH][FAISS] candidates=%d",
        len(candidate_groups),
    )

    # --------------------------------------------------
    # 3️⃣ 후보 그룹만 점수 집계 (max 기준)
    # --------------------------------------------------
    scores = []

    for group_id, sims in candidate_groups.items():
        if group_id not in groups:
            continue

        max_score = max(sims)
        avg_score = sum(sims) / len(sims)

        logger.info(
            "[GROUP_SEARCH][GROUP_SUMMARY] group=%s max=%.4f avg=%.4f",
            group_id,
            max_score,
            avg_score,
        )

        scores.append({
            "group_id": group_id,
            "score": max_score,
        })

    if not scores:
        logger.info("[GROUP_SEARCH][RESULT] no candidates")
        return {
            "matched": False,
            "group_id": None,
            "score": -1.0,
        }

    # --------------------------------------------------
    # 4️⃣ 자동 threshold 튜닝
    # --------------------------------------------------
    scores.sort(key=lambda x: x["score"], reverse=True)

    best = scores[0]
    second = scores[1] if len(scores) > 1 else None

    gap = best["score"] - (second["score"] if second else 0.0)

    matched = (
        best["score"] >= MIN_THRESHOLD and
        gap >= GAP_THRESHOLD
    )

    logger.info(
        "[GROUP_SEARCH][DECISION] matched=%s group=%s score=%.4f gap=%.4f "
        "min_th=%.2f gap_th=%.2f",
        matched,
        best["group_id"],
        best["score"],
        gap,
        MIN_THRESHOLD,
        GAP_THRESHOLD,
    )

    return {
        "matched": matched,
        "group_id": best["group_id"] if matched else None,
        "score": best["score"],
    }