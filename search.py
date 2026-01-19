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
# 🔥 사용자 파우치 그룹 검색 (Node 연동용)
# ==================================================

def search_image_with_groups(
    image_path: str,
    groups: dict,
):
    """
    image_path: 업로드된 이미지 경로
    groups: {
        "12": ["s3Key1", "s3Key2"],
        "15": ["s3Key3", "s3Key4"]
    }

    return:
    {
        "matched": bool,
        "group_id": str | None,
        "score": float | None
    }
    """

    logger.info(
        "[GROUP_SEARCH][START] groups=%d image=%s",
        len(groups),
        os.path.basename(image_path),
    )

    # ------------------------------
    # 1️⃣ query embedding
    # ------------------------------
    img = Image.open(image_path).convert("RGB")
    q = image_to_vector(img)

    best_group_id = None
    best_score = -1.0

    # ------------------------------
    # 2️⃣ 그룹별 평균 벡터와 cosine similarity
    # ------------------------------
    for group_id, image_keys in groups.items():
        vectors = []

        for key in image_keys:
            try:
                # ⚠️ build_index와 동일한 방식으로
                # S3 이미지들은 이미 index에 반영됨
                # 여기서는 group_id 기준 비교만 수행
                idxs = np.where(PRODUCT_IDS == str(group_id))[0]
                if len(idxs) == 0:
                    continue

                vec = INDEX.reconstruct(int(idxs[0]))
                vectors.append(vec)

            except Exception as e:
                logger.warning(
                    "[GROUP_SEARCH][WARN] group=%s key=%s error=%s",
                    group_id,
                    key,
                    str(e),
                )

        if not vectors:
            logger.debug(
                "[GROUP_SEARCH][SKIP] group=%s no vectors",
                group_id,
            )
            continue

        group_vec = np.mean(vectors, axis=0)
        group_vec = group_vec / np.linalg.norm(group_vec)

        score = float(np.dot(q, group_vec))

        logger.debug(
            "[GROUP_SEARCH][CANDIDATE] group=%s score=%.4f",
            group_id,
            score,
        )

        if score > best_score:
            best_score = score
            best_group_id = group_id

    # ------------------------------
    # 3️⃣ 판정
    # ------------------------------
    matched = best_score >= SIMILARITY_THRESHOLD

    logger.info(
        "[GROUP_SEARCH][RESULT] matched=%s group=%s score=%.4f threshold=%.2f",
        matched,
        best_group_id,
        best_score,
        SIMILARITY_THRESHOLD,
    )

    if not matched:
        return {
            "matched": False,
            "group_id": None,
            "score": best_score,
        }

    return {
        "matched": True,
        "group_id": best_group_id,
        "score": best_score,
    }
