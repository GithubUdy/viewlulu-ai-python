"""
search.py (FINAL – SPEED OPTIMIZED)
--------------------------------------------------
✅ FAISS index preload (startup)
✅ SigLIP embedding (query 1회만)
✅ FAISS reconstruct() 기반 그룹 비교 (🔥 재임베딩 제거)
✅ 1:N (화장품 1 : 이미지 N) max-score 전략
✅ cosine similarity
✅ 자동 threshold 튜닝 (min + gap)
✅ Node 서버 연동 응답 구조 유지
✅ 🔥 상세 로그 + 타이밍 로그
"""

import os
import logging
import time
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
SIMILARITY_THRESHOLD = 0.3
FAISS_TOP_K = 5

# 🔥 자동 튜닝 기준
MIN_THRESHOLD = 0.45
GAP_THRESHOLD = 0.07


# ==================================================
# Load Assets (startup 1회)
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

    logger.info("[FAISS] index loaded (ntotal=%d)", index.ntotal)
    return index, product_ids


INDEX, PRODUCT_IDS = _load_assets()


# ==================================================
# (기존 유지) 전체 DB 검색 ❌ 변경 금지
# ==================================================
def search_image(image_path: str, top_k: int):
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
        return {"matched": False, "best": None, "results": []}

    best = results[0]

    if best["similarity"] < SIMILARITY_THRESHOLD:
        return {"matched": False, "best": best, "results": results}

    return {"matched": True, "best": best, "results": results}

def _embed_image_path(image_path: str):
    img = Image.open(image_path).convert("RGB")
    vec = image_to_vector(img)
    vec = vec / np.linalg.norm(vec)
    return vec.reshape(1, -1)



# ==================================================
# 🔥 사용자 파우치 그룹 검색 (최종 핵심)
# ==================================================
def search_image_with_groups(image_path: str, groups: dict):
    """
    image_path: 촬영 이미지 경로
    groups: {
        "12": ["/tmp/12/1.jpg", "..."],
        ...
    }
    """

    t0 = time.perf_counter()

    logger.info(
        "[GROUP_SEARCH][START] groups=%d image=%s",
        len(groups),
        os.path.basename(image_path),
    )

    if not groups:
        return {"matched": False, "group_id": None, "score": -1.0}

    # --------------------------------------------------
    # 1️⃣ Query embedding (🔥 단 1회)
    # --------------------------------------------------
    q = _embed_image_path(image_path)

    # --------------------------------------------------
    # 2️⃣ FAISS 후보 그룹 검색 (🔥 핵심)
    # --------------------------------------------------
    q2 = q.reshape(1, -1)
    sims, idxs = INDEX.search(q2, min(FAISS_TOP_K, INDEX.ntotal))

    candidate_group_ids = []
    for idx in idxs[0]:
        if idx < 0:
            continue
        gid = str(PRODUCT_IDS[int(idx)])
        if gid in groups:
            candidate_group_ids.append(gid)

    logger.info(
        "[GROUP_SEARCH][FAISS] candidates=%s",
        candidate_group_ids,
    )

    if not candidate_group_ids:
        logger.info("[GROUP_SEARCH][RESULT] no candidates")
        return {"matched": False, "group_id": None, "score": -1.0}

    # --------------------------------------------------
    # 3️⃣ 후보 그룹만 1:4 비교
    # --------------------------------------------------
    group_scores = []

    for group_id in candidate_group_ids:
        image_paths = groups.get(group_id, [])
        scores = []

        for img_path in image_paths:
            try:
                v = _embed_image_path(img_path)
                sim = float(np.dot(q, v))
                scores.append(sim)
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
        return {"matched": False, "group_id": None, "score": -1.0}

    # --------------------------------------------------
    # 4️⃣ 자동 튜닝 판정 (기존 유지)
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

    t1 = time.perf_counter()

    logger.info(
        "[GROUP_SEARCH][DECISION] matched=%s best=%s score=%.4f gap=%.4f total_ms=%.1f",
        matched,
        best["group_id"],
        best_score,
        gap,
        (t1 - t0) * 1000,
    )

    return {
        "matched": matched,
        "group_id": best["group_id"] if matched else None,
        "score": best_score,
    }
