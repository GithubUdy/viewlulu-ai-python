import os
import uuid
import logging
import json
from typing import Dict, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException

logging.basicConfig(level=logging.INFO)

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
    load_model()   # 🔥 서버 시작 시 1회만 실행


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
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}",
        )

    fname = f"{uuid.uuid4()}.{ext}"
    path = os.path.join(UPLOAD_DIR, fname)

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")

        with open(path, "wb") as f:
            f.write(content)

        from search import search_image
        results = search_image(path, top_k=5)

        if not results["matched"]:
            return {
                "matched": False,
                "message": "일치하는 화장품을 찾지 못했습니다.",
                "bestDistance": (
                    results["best"]["distance"]
                    if results.get("best")
                    else None
                ),
            }

        return {
            "matched": True,
            "detectedId": results["best"]["product_id"],
            "bestDistance": results["best"]["distance"],
        }

    except HTTPException:
        raise

    except Exception as e:
        logging.exception("Unexpected error during pouch_search")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(path):
            os.remove(path)


# ==================================================
# 🔥 신규: 파우치 그룹 전용 검색 (정답 구조)
# ==================================================
@app.post("/pouch/group-search")
async def pouch_group_search(
    file: UploadFile = File(...),
    groups: str = Form(...),
):
    """
    groups (JSON string):
    {
      "12": ["/tmp/a.jpg", "/tmp/b.jpg"],
      "15": ["/tmp/c.jpg", "/tmp/d.jpg"]
    }
    """

    if not file:
        raise HTTPException(status_code=400, detail="file is required")

    try:
        group_dict: Dict[str, List[str]] = json.loads(groups)
    except Exception:
        raise HTTPException(status_code=400, detail="groups must be valid JSON")

    filename = file.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    fname = f"{uuid.uuid4()}.{ext}"
    path = os.path.join(UPLOAD_DIR, fname)

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")

        with open(path, "wb") as f:
            f.write(content)

        from search import search_image_with_groups

        result = search_image_with_groups(
            image_path=path,
            groups=group_dict,
        )

        if not result["matched"]:
            return {
                "matched": False,
                "message": "일치하는 화장품을 찾지 못했습니다.",
            }

        return {
            "matched": True,
            "detectedGroupId": result["group_id"],
            "score": result["score"],
        }

    except HTTPException:
        raise

    except Exception as e:
        logging.exception("Unexpected error during pouch_group_search")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(path):
            os.remove(path)
