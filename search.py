"""
search.py (EC2 배포 안정 최종본)
--------------------------------------------------
✅ FAISS 인덱스 로드
✅ product_ids 로드
✅ search_image(image_path, top_k) 제공
✅ 실행 위치 달라도 index 경로 안정적으로 동작
"""

import os
import faiss
import numpy as np
from PIL import Image

from siglip import image_to_vector

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INDEX_PATH = os.path.join(BASE_DIR, "index", "siglip.index")
IDS_PATH = os.path.join(BASE_DIR, "index", "product_ids.npy")


def _load_assets():
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found: {INDEX_PATH}")
    if not os.path.exists(IDS_PATH):
        raise FileNotFoundError(f"product_ids not found: {IDS_PATH}")

    index = faiss.read_index(INDEX_PATH)
    product_ids = np.load(IDS_PATH, allow_pickle=True)
    return index, product_ids


# ✅ 서버 프로세스 시작 시 1회 로드 (성능/안정성)
INDEX, PRODUCT_IDS = _load_assets()


def search_image(image_path: str, top_k: int = 5):
    """
    업로드된 이미지(path)를 받아
    FAISS에서 가장 유사한 화장품을 검색

    return:
        [
            {"product_id": str, "score": float},
            ...
        ]
    """

    if top_k <= 0:
        top_k = 5

    # 1) 이미지 로드
    img = Image.open(image_path).convert("RGB")

    # 2) SigLIP 임베딩
    q = image_to_vector(img).reshape(1, -1)

    # 3) FAISS 검색
    sims, idxs = INDEX.search(q, top_k)

    # 4) 결과 정리
    results = []
    for score, idx in zip(sims[0], idxs[0]):
        # idx가 -1로 나올 수도 있어서 방어
        if int(idx) < 0:
            continue
        pid = PRODUCT_IDS[int(idx)]
        results.append({"product_id": str(pid), "score": float(score)})

    return results
