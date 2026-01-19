"""
search.py (FINAL STABLE)
--------------------------------------------------
✅ FAISS index preload (startup)
✅ SigLIP embedding 사용
✅ 그룹 평균 벡터 기반 검색
✅ cosine similarity 기준
✅ threshold 기반 매칭 판정
✅ Node 서버 연동용 안정 응답 구조
✅ 🔥 group-search 상세 로그 추가 (후보 / score / 판정)
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
# Search Config (🔥 실서비스 기준)
# ==================================================

# cosine similarity 기준
SIMILARITY_THRESHOLD = 0.3
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

    logger.info(
        "[FAISS] index loaded (total_groups=%d)",
        index.ntotal,
    )

    return index, product_ids


# 🔥 서버 시작 시 1회 로드
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
# Logger
# ==================================================
logger = logging.getLogger(__name__)


# ==================================================
# Threshold Config (🔥 핵심)
# ==================================================

# 절대 최소 점수 (이보다 낮으면 무조건 실패)
MIN_THRESHOLD = 0.45

# 1등과 2등 점수 차이 (확신도)
GAP_THRESHOLD = 0.07


# ==================================================
# 🔥 사용자 파우치 그룹 검색 (Node 연동용)
# ==================================================
def search_image_with_groups(image_path: str, groups: dict):
    """
    image_path: 촬영 이미지 경로
    groups: {
        "12": ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"],
        ...
    }

    return:
    {
        "matched": bool,
        "group_id": str | None,
        "score": float
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
    q = image_to_vector(img)
    q = q / np.linalg.norm(q)

    group_scores = []

    # --------------------------------------------------
    # 2️⃣ 그룹별 1:4 비교 (max 기준)
    # --------------------------------------------------
    for group_id, image_paths in groups.items():
        scores = []

        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                v = image_to_vector(img)
                v = v / np.linalg.norm(v)

                sim = float(np.dot(q, v))
                scores.append(sim)

                logger.debug(
                    "[GROUP_SEARCH][SCORE] group=%s img=%s sim=%.4f",
                    group_id,
                    os.path.basename(img_path),
                    sim,
                )

            except Exception as e:
                logger.warning(
                    "[GROUP_SEARCH][IMAGE_FAIL] group=%s img=%s err=%s",
                    group_id,
                    img_path,
                    str(e),
                )

        if not scores:
            continue

        max_score = max(scores)
        avg_score = sum(scores) / len(scores)

        logger.info(
            "[GROUP_SEARCH][GROUP_SUMMARY] group=%s max=%.4f avg=%.4f",
            group_id,
            max_score,
            avg_score,
        )

        group_scores.append({
            "group_id": group_id,
            "max": max_score,
        })

    if not group_scores:
        logger.info("[GROUP_SEARCH][RESULT] no valid groups")
        return {
            "matched": False,
            "group_id": None,
            "score": -1.0,
        }

    # --------------------------------------------------
    # 3️⃣ 자동 튜닝 판정 (🔥 핵심)
    # --------------------------------------------------
    group_scores.sort(key=lambda x: x["max"], reverse=True)

    best = group_scores[0]
    second = group_scores[1] if len(group_scores) > 1 else None

    best_score = best["max"]
    gap = best_score - (second["max"] if second else 0.0)

    matched = (
        best_score >= MIN_THRESHOLD and
        gap >= GAP_THRESHOLD
    )

    logger.info(
        "[GROUP_SEARCH][DECISION] matched=%s best=%s score=%.4f gap=%.4f "
        "min_th=%.2f gap_th=%.2f",
        matched,
        best["group_id"],
        best_score,
        gap,
        MIN_THRESHOLD,
        GAP_THRESHOLD,
    )

    return {
        "matched": matched,
        "group_id": best["group_id"] if matched else None,
        "score": best_score,
    }