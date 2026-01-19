import os
import uuid
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException

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
# Image search endpoint
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

        # --------------------------------------------------
        # ❗ 매칭 실패도 정상 응답 (200 OK)
        # --------------------------------------------------
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

        # --------------------------------------------------
        # ✅ 매칭 성공
        # --------------------------------------------------
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
