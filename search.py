"""
search.py (FINAL – SPEED OPTIMIZED + CACHE)
--------------------------------------------------
✅ FAISS index preload (startup)
✅ SigLIP embedding (query 1회만)
✅ 1:N (화장품 1 : 이미지 N) max-score 전략
✅ cosine similarity
✅ 자동 threshold 튜닝 (min + gap)
✅ Node 서버 연동 응답 구조 유지
✅ 🔥 FAISS 후보축소(없으면 fallback)
✅ 🔥 (A안) 경로가 바뀌어도 먹는 SHA1 임베딩 캐시
"""

import os
import logging
import time
import hashlib
import numpy as np
import faiss
from PIL import Image
from typing import Dict, List, Tuple

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

# FAISS 후보 그룹 수(후보 축소용; empty면 fallback)
FAISS_TOP_K = 5

# 🔥 자동 튜닝 기준 (유지)
MIN_THRESHOLD = 0.45
GAP_THRESHOLD = 0.07


# ==================================================
# In-memory Cache (A안 핵심)
# ==================================================
# 이미지 내용(sha1) -> embedding (D,)
_EMBED_CACHE: Dict[str, np.ndarray] = {}

# group_id -> (image_hashes_tuple, stacked_vectors (N,D))
_GROUP_CACHE: Dict[str, Tuple[Tuple[str, ...], np.ndarray]] = {}


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


# ==================================================
# Helpers
# ==================================================
def _file_sha1(path: str) -> str:
    """파일 경로가 바뀌어도 동일 이미지면 같은 캐시 키가 되도록 내용 해시 사용"""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _embed_image_path(image_path: str) -> np.ndarray:
    """
    - (A안) sha1 기반 임베딩 캐시 적용
    - 출력 shape: (D,) float32 L2-normalized
    """
    key = _file_sha1(image_path)
    cached = _EMBED_CACHE.get(key)
    if cached is not None:
        return cached

    img = Image.open(image_path).convert("RGB")
    vec = image_to_vector(img).astype("float32").reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm

    _EMBED_CACHE[key] = vec
    return vec


def _get_group_matrix(group_id: str, image_paths: List[str]) -> np.ndarray:
    """
    group_id의 이미지들에 대해 (N,D) 행렬을 반환.
    - group 캐시: (이미지 해시 튜플이 같으면) 재계산 없이 재사용
    """
    hashes = tuple(_file_sha1(p) for p in image_paths)
    cached = _GROUP_CACHE.get(group_id)

    if cached is not None:
        cached_hashes, cached_mat = cached
        if cached_hashes == hashes:
            return cached_mat

    vectors = []
    for p in image_paths:
        v = _EMBED_CACHE.get(_file_sha1(p))
        if v is None:
            v = _embed_image_path(p)
        vectors.append(v)

    mat = np.stack(vectors, axis=0).astype("float32")  # (N,D)
    _GROUP_CACHE[group_id] = (hashes, mat)
    return mat


# ==================================================
# 🔥 사용자 파우치 그룹 검색 (FINAL)
# - FAISS = 후보 축소용 (threshold ❌)
# - 정확도 판단 = 그룹 내부 비교 (max-score)
# - FAISS 후보 없으면 전체 그룹 fallback
# - (A안) sha1 기반 캐시로 재임베딩 제거
# ==================================================
def search_image_with_groups(image_path: str, groups: dict):
    """
    image_path: 촬영 이미지 경로
    groups: {
        "12": ["/tmp/12/1.jpg", "..."],
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
        return {"matched": False, "group_id": None, "score": -1.0}

    # --------------------------------------------------
    # 1️⃣ Query embedding (🔥 단 1회 + 캐시)
    # --------------------------------------------------
    tq0 = time.perf_counter()
    q = _embed_image_path(image_path)  # (D,)
    tq1 = time.perf_counter()
    logger.info("[GROUP_SEARCH][TIME] query_embed_ms=%.1f", (tq1 - tq0) * 1000)

    # --------------------------------------------------
    # 2️⃣ FAISS 후보 그룹 검색 (후보 축소 ONLY)
    # --------------------------------------------------
    candidate_group_ids: List[str] = []

    if INDEX is not None and INDEX.ntotal > 0:
        q2 = q.reshape(1, -1)  # (1,D)
        sims, idxs = INDEX.search(q2, min(FAISS_TOP_K, INDEX.ntotal))

        for idx in idxs[0]:
            if idx < 0:
                continue
            gid = str(PRODUCT_IDS[int(idx)])
            if gid in groups and gid not in candidate_group_ids:
                candidate_group_ids.append(gid)

    # 후보 없으면 전체 fallback
    if not candidate_group_ids:
        logger.info("[GROUP_SEARCH][FAISS] empty → fallback to all groups")
        candidate_group_ids = list(groups.keys())
    else:
        logger.info("[GROUP_SEARCH][FAISS] candidates=%s", candidate_group_ids)

    # --------------------------------------------------
    # 3️⃣ 그룹 내부 비교 (max-score) — 캐시된 벡터로 dot만 수행
    # --------------------------------------------------
    group_scores = []

    for group_id in candidate_group_ids:
        image_paths = groups.get(group_id, [])
        if not image_paths:
            continue

        tg0 = time.perf_counter()
        mat = _get_group_matrix(group_id, image_paths)      # (N,D)
        sims = mat @ q                                      # (N,)
        max_score = float(np.max(sims))
        avg_score = float(np.mean(sims))
        tg1 = time.perf_counter()

        logger.info(
            "[GROUP_SEARCH][GROUP_SUMMARY] group=%s max=%.4f avg=%.4f imgs=%d ms=%.1f",
            group_id,
            max_score,
            avg_score,
            mat.shape[0],
            (tg1 - tg0) * 1000,
        )

        group_scores.append({"group_id": group_id, "max": max_score})

    if not group_scores:
        return {"matched": False, "group_id": None, "score": -1.0}

    # --------------------------------------------------
    # 4️⃣ 자동 튜닝 판정 (min + gap) 유지
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
        "min_th=%.2f gap_th=%.2f total_ms=%.1f",
        matched,
        best["group_id"],
        best_score,
        second_score,
        gap,
        MIN_THRESHOLD,
        GAP_THRESHOLD,
        (t1 - t0) * 1000,
    )

    return {
        "matched": matched,
        "group_id": best["group_id"] if matched else None,
        "score": best_score,
    }
