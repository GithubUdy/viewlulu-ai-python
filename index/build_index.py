"""
build_index.py (FINAL)
--------------------------------------------------
✅ S3에서 화장품 그룹 이미지 로드
✅ SigLIP embedding 추출
✅ 그룹 단위 평균 벡터 생성
✅ FAISS index (product-level) 생성
✅ product_ids.npy 저장
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

# SigLIP embedding dim (open-clip ViT-B/16 = 512)
EMBED_DIM = 512

# ==================================================
# AWS S3 Client
# ==================================================

s3 = boto3.client("s3", region_name=AWS_REGION)


# ==================================================
# Helpers
# ==================================================

def list_all_group_images():
    """
    S3 전체를 스캔해서
    group_id -> [image_keys] 형태로 반환
    """
    paginator = s3.get_paginator("list_objects_v2")

    groups = {}

    for page in paginator.paginate(Bucket=S3_BUCKET):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            # users/{userId}/cosmetics/{groupId}/{filename}
            parts = key.split("/")
            if len(parts) < 5:
                continue

            if parts[2] != "cosmetics":
                continue

            group_id = parts[3]
            ext = key.lower().split(".")[-1]
            if ext not in {"jpg", "jpeg", "png", "webp"}:
                continue

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
    print("🔹 Loading SigLIP model...")
    load_model()  # 🔥 1회 로드

    print("🔹 Scanning S3 for cosmetic groups...")
    groups = list_all_group_images()

    if not groups:
        raise RuntimeError("No cosmetic images found in S3")

    print(f"✅ Found {len(groups)} cosmetic groups")

    vectors = []
    product_ids = []

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

        # 🔥 핵심: 그룹 평균 벡터
        group_vector = np.mean(embeddings, axis=0)
        group_vector = group_vector / np.linalg.norm(group_vector)

        vectors.append(group_vector)
        product_ids.append(group_id)

    if not vectors:
        raise RuntimeError("No valid group vectors created")

    vectors = np.vstack(vectors).astype("float32")

    print("🔹 Creating FAISS index...")
    index = faiss.IndexFlatIP(EMBED_DIM)  # cosine similarity
    index.add(vectors)

    print("🔹 Saving index files...")
    faiss.write_index(index, INDEX_PATH)
    np.save(IDS_PATH, np.array(product_ids))

    print("🎉 DONE")
    print(f"- index saved to: {INDEX_PATH}")
    print(f"- product ids saved to: {IDS_PATH}")
    print(f"- total products indexed: {len(product_ids)}")


if __name__ == "__main__":
    build_index()
