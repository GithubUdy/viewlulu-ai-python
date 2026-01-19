"""
app.py (FINAL STABLE)
--------------------------------------------------
✅ /pouch/search 기존 유지 (전체 DB 검색, 검증용)
✅ /pouch/group-search 사용자 파우치 기준 검색
✅ XML / HTML / 깨진 파일 방어 (imghdr)
✅ SigLIP startup preload (1회)
✅ 업로드 / 판정 / 결과 로그 출력
✅ search.py / siglip.py 최종본과 완전 호환
"""

import os
import uuid
import json
import imghdr
import logging
from typing import Dict, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException

# ==================================================
# Logging
# ==================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# ==================================================
# App
# ==================================================
app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "tmp")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXT = {"jpg", "jpeg", "png", "webp"}

# ==================================================
# Startup: preload SigLIP model (1회)
# ==================================================
@app.on_event("startup")
def preload_models():
    from siglip import load_model
    load_model()
    logger.info("[STARTUP] SigLIP model preloaded")

# ==================================================
# Health check
# ==================================================
@app.get("/")
def health():
    return {"status": "ok"}

# ==================================================
# (기존 유지) 전체 DB 기반 검색
# ==================================================
@app.post("/pouch/search")
async def pouch_search(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="file is required")

    filename = file.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    tmp_name = f"{uuid.uuid4()}.{ext}"
    tmp_path = os.path.join(UPLOAD_DIR, tmp_name)

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")

        with open(tmp_path, "wb") as f:
            f.write(content)

        logger.info("[UPLOAD][SEARCH] filename=%s size=%d", filename, len(content))

        kind = imghdr.what(tmp_path)
        logger.info("[IMAGE_CHECK][SEARCH] kind=%s", kind)

        if kind is None:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")

        from search import search_image
        results = search_image(tmp_path, top_k=5)

        best = results.get("best")

        logger.info(
            "[RESULT][SEARCH] matched=%s product_id=%s similarity=%.4f distance=%.4f",
            results["matched"],
            best["product_id"] if best else None,
            best["similarity"] if best else -1.0,
            best["distance"] if best else -1.0,
        )

        if not results["matched"]:
            return {
                "matched": False,
                "message": "일치하는 화장품을 찾지 못했습니다.",
                "bestDistance": best["distance"] if best else None,
            }

        return {
            "matched": True,
            "detectedId": best["product_id"],
            "bestDistance": best["distance"],
        }

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ==================================================
# 🔥 사용자 파우치 그룹 전용 검색 (최종 / 수정 완료)
# ==================================================
@app.post("/pouch/group-search")
async def pouch_group_search(
    file: UploadFile = File(...),
    groups: str = Form(...),   # ✅ 반드시 Form
):
    if not file:
        raise HTTPException(status_code=400, detail="file is required")

    try:
        group_dict: Dict[str, List[str]] = json.loads(groups)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"groups must be valid JSON string ({e})"
        )

    logger.info(
        "[GROUP_SEARCH][REQUEST_OK] groups=%d file=%s",
        len(group_dict),
        file.filename,
    )

    filename = file.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    tmp_name = f"{uuid.uuid4()}.{ext}"
    tmp_path = os.path.join(UPLOAD_DIR, tmp_name)

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")

        with open(tmp_path, "wb") as f:
            f.write(content)

        logger.info(
            "[UPLOAD][GROUP] filename=%s size=%d groups=%d",
            filename,
            len(content),
            len(group_dict),
        )

        kind = imghdr.what(tmp_path)
        logger.info("[IMAGE_CHECK][GROUP] kind=%s", kind)

        if kind is None:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")

        from search import search_image_with_groups

        result = search_image_with_groups(
            image_path=tmp_path,
            groups=group_dict,
        )

        logger.info(
            "[RESULT][GROUP] matched=%s group_id=%s score=%.4f",
            result["matched"],
            result.get("group_id"),
            result.get("score", -1.0),
        )

        if not result["matched"]:
            return {
                "matched": False,
                "message": "일치하는 화장품을 찾지 못했습니다.",
                "score": result.get("score"),
            }

        return {
            "matched": True,
            "detectedGroupId": result["group_id"],
            "score": result["score"],
        }

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
