"""
build_index.py (FINAL STABLE)
--------------------------------------------------
✅ S3에서 화장품 그룹 이미지 로드
✅ SigLIP embedding 추출
✅ 그룹 단위 평균 벡터 생성 (1:N → 1)
✅ FAISS IndexFlatIP (cosine similarity)
✅ product_ids.npy 저장
✅ search.py / siglip.py 와 완전 호환
"""

import os
import io
import boto3
import faiss
import numpy as np
from PIL import Image
from tqdm import tqdm

from siglip import load_model, image_to_vector


# ==================================================
# Config
# ==================================================

AWS_REGION = "ap-northeast-2"
S3_BUCKET = "viewlulus3"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(BASE_DIR, "index")
os.makedirs(INDEX_DIR, exist_ok=True)

INDEX_PATH = os.path.join(INDEX_DIR, "siglip.index")
IDS_PATH = os.path.join(INDEX_DIR, "product_ids.npy")

# SigLIP embedding dim (ViT-B/16 = 512)
EMBED_DIM = 512

VALID_EXT = {"jpg", "jpeg", "png", "webp"}


# ==================================================
# AWS S3 Client
# ==================================================

s3 = boto3.client("s3", region_name=AWS_REGION)


# ==================================================
# Helpers
# ==================================================

def list_all_group_images():
    """
    S3 전체 스캔
    users/{userId}/cosmetics/{groupId}/{filename}
    →
    { groupId: [s3Key, ...] }
    """
    paginator = s3.get_paginator("list_objects_v2")
    groups: dict[str, list[str]] = {}

    for page in paginator.paginate(Bucket=S3_BUCKET):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            parts = key.split("/")
            if len(parts) < 5:
                continue

            # users/{userId}/cosmetics/{groupId}/...
            if parts[2] != "cosmetics":
                continue

            ext = key.lower().split(".")[-1]
            if ext not in VALID_EXT:
                continue

            group_id = parts[3]
            groups.setdefault(group_id, []).append(key)

    return groups


def load_image_from_s3(key: str) -> Image.Image:
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    data = obj["Body"].read()
    return Image.open(io.BytesIO(data)).convert("RGB")


# ==================================================
# Main
# ==================================================

def build_index():
    print("🔹 Loading SigLIP model (1-time preload)...")
    load_model()

    print("🔹 Scanning S3 for cosmetic groups...")
    groups = list_all_group_images()

    if not groups:
        raise RuntimeError("No cosmetic images found in S3")

    print(f"✅ Found {len(groups)} cosmetic groups")

    vectors: list[np.ndarray] = []
    product_ids: list[str] = []

    for group_id, image_keys in tqdm(groups.items(), desc="Building index"):
        embeddings = []

        for key in image_keys:
            try:
                img = load_image_from_s3(key)
                vec = image_to_vector(img)  # (512,)
                embeddings.append(vec)
            except Exception as e:
                print(f"[WARN] Failed image: {key} ({e})")

        if not embeddings:
            print(f"[SKIP] group {group_id} has no valid images")
            continue

        # 🔥 핵심: 그룹 단위 평균 벡터 (1:N → 1)
        group_vec = np.mean(embeddings, axis=0).astype("float32")
        norm = np.linalg.norm(group_vec)
        if norm > 0:
            group_vec /= norm

        vectors.append(group_vec)
        product_ids.append(str(group_id))

    if not vectors:
        raise RuntimeError("No valid group vectors created")

    vectors_np = np.vstack(vectors).astype("float32")

    print("🔹 Creating FAISS index (IndexFlatIP)...")
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vectors_np)

    print("🔹 Saving index files...")
    faiss.write_index(index, INDEX_PATH)
    np.save(IDS_PATH, np.array(product_ids, dtype=object))

    print("🎉 DONE")
    print(f"- index saved to: {INDEX_PATH}")
    print(f"- product ids saved to: {IDS_PATH}")
    print(f"- total groups indexed: {len(product_ids)}")


if __name__ == "__main__":
    build_index()
