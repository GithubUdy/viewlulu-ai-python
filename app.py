import os
import uuid
import logging
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException

logging.basicConfig(level=logging.INFO)

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "tmp")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXT = {"jpg", "jpeg", "png", "webp"}


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/pouch/search")
async def pouch_search(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="file is required")

    filename = file.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in ALLOWED_EXT:
        logging.error(f"❌ Unsupported extension: '{ext}', filename='{filename}'")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension: '{ext}'",
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

        if not results:
            return {"top1": None, "top5": []}

        return {"top1": results[0], "top5": results}

    except HTTPException as he:
        logging.error(f"[HTTPException] {he.detail}")
        raise he

    except Exception as e:
        logging.error("🔥 AI SEARCH ERROR")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(path):
            os.remove(path)
