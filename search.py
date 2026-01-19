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
✅ 🔥 (추가) group-search 미세 최적화 + 타이밍 로그
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
# Search Config (기존 유지)
# ==================================================
SIMILARITY_THRESHOLD = 0.3
TOP_K = 5


# ==================================================
# Threshold Config (🔥 자동 튜닝)
# ==================================================
MIN_THRESHOLD = 0.45
GAP_THRESHOLD = 0.07


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

    logger.info("[FAISS] index loaded (total_groups=%d)", index.ntotal)
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
        return {"matched": False, "best": None, "results": []}

    best = results[0]

    if best["similarity"] < SIMILARITY_THRESHOLD:
        return {"matched": False, "best": best, "results": results}

    return {"matched": True, "best": best, "results": results}


# ==================================================
# 내부 유틸 (미세 최적화)
# ==================================================
def _l2_normalize(v: np.ndarray) -> np.ndarray:
    # np.linalg.norm은 내부적으로 sqrt(dot)인데,
    # float32 기준으로 아래가 미세하게 더 빠른 경우가 많음.
    denom = float(np.sqrt(np.dot(v, v)))  # type: ignore
    if denom == 0.0:
        return v
    return v / denom


def _embed_image_path(img_path: str) -> np.ndarray:
    """
    - PIL open/convert 안정화
    - image_to_vector 호출 후 float32 정규화
    """
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        v = image_to_vector(im).astype("float32")
    return _l2_normalize(v)


# ==================================================
# 🔥 사용자 파우치 그룹 검색 (Node 연동용)
# ==================================================
def search_image_with_groups(image_path: str, groups: dict):
    """
    image_path: 촬영 이미지 경로
    groups: {
        "12": ["/tmp/12/img1.jpg", "/tmp/12/img2.jpg", "/tmp/12/img3.jpg", "/tmp/12/img4.jpg"],
        ...
    }

    return:
    {
        "matched": bool,
        "group_id": str | None,
        "score": float
    }
    """

    t0 = time.perf_counter()

    logger.info(
        "[GROUP_SEARCH][START] groups=%d image=%s",
        len(groups),
        os.path.basename(image_path),
    )

    if not groups:
        logger.info("[GROUP_SEARCH][RESULT] empty groups")
        return {"matched": False, "group_id": None, "score": -1.0}

    # --------------------------------------------------
    # 1️⃣ Query embedding (1회)
    # --------------------------------------------------
    tq0 = time.perf_counter()
    q = _embed_image_path(image_path)
    tq1 = time.perf_counter()
    logger.info("[GROUP_SEARCH][TIME] query_embed_ms=%.1f", (tq1 - tq0) * 1000)

    # --------------------------------------------------
    # 2️⃣ 그룹별 1:4 비교 (max 기준)
    # --------------------------------------------------
    group_scores = []
    embed_count = 0
    failed_count = 0

    for group_id, image_paths in groups.items():
        if not image_paths:
            continue

        scores = []

        # group 단위 타이밍(원인 추적용)
        tg0 = time.perf_counter()

        for img_path in image_paths:
            try:
                v = _embed_image_path(img_path)
                embed_count += 1

                sim = float(np.dot(q, v))
                scores.append(sim)

                logger.debug(
                    "[GROUP_SEARCH][SCORE] group=%s img=%s sim=%.4f",
                    group_id,
                    os.path.basename(img_path),
                    sim,
                )

            except Exception as e:
                failed_count += 1
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

        tg1 = time.perf_counter()
        logger.info(
            "[GROUP_SEARCH][GROUP_SUMMARY] group=%s max=%.4f avg=%.4f imgs=%d ms=%.1f",
            group_id,
            max_score,
            avg_score,
            len(scores),
            (tg1 - tg0) * 1000,
        )

        group_scores.append({"group_id": str(group_id), "max": float(max_score)})

    if not group_scores:
        logger.info(
            "[GROUP_SEARCH][RESULT] no valid groups embed_count=%d failed=%d",
            embed_count,
            failed_count,
        )
        return {"matched": False, "group_id": None, "score": -1.0}

    # --------------------------------------------------
    # 3️⃣ 자동 튜닝 판정 (min + gap)
    # --------------------------------------------------
    group_scores.sort(key=lambda x: x["max"], reverse=True)

    best = group_scores[0]
    second = group_scores[1] if len(group_scores) > 1 else None

    best_score = best["max"]
    second_score = second["max"] if second else 0.0
    gap = best_score - second_score

    matched = (best_score >= MIN_THRESHOLD) and (gap >= GAP_THRESHOLD)

    t1 = time.perf_counter()
    logger.info(
        "[GROUP_SEARCH][DECISION] matched=%s best=%s score=%.4f second=%.4f gap=%.4f "
        "min_th=%.2f gap_th=%.2f total_ms=%.1f embed_count=%d failed=%d",
        matched,
        best["group_id"],
        best_score,
        second_score,
        gap,
        MIN_THRESHOLD,
        GAP_THRESHOLD,
        (t1 - t0) * 1000,
        embed_count,
        failed_count,
    )

    return {
        "matched": matched,
        "group_id": best["group_id"] if matched else None,
        "score": float(best_score),
    }
