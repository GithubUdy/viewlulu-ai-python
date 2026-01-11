"""
app.py (EC2 배포 안정 최종본)
--------------------------------------------------
✅ FastAPI 서버
✅ / health check
✅ /pouch/search : 이미지 업로드 -> tmp 저장 -> 검색 -> 삭제 -> 결과 반환
✅ EC2 환경에서 작업 디렉토리 달라도 경로 안정적으로 동작
✅ 임시 파일 확장자/파일명 안전 처리
"""

import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException

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
    """
    RN/Node에서 multipart/form-data로 이미지를 올리면
    top1, top5 결과를 반환한다.
    """
    # ✅ 요청 파일 검증
    if not file:
        raise HTTPException(status_code=400, detail="file is required")

    filename = file.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXT:
        # RN 카메라는 보통 jpg/png라서 실사용 문제 없음.
        # 그래도 서버 안정성을 위해 제한.
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension: '{ext}'. Allowed: {sorted(ALLOWED_EXT)}",
        )

    # ✅ 임시 저장 경로
    fname = f"{uuid.uuid4()}.{ext}"
    path = os.path.join(UPLOAD_DIR, fname)

    # ✅ 파일 저장
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")

        with open(path, "wb") as f:
            f.write(content)

        # 🔥 핵심: 검색 실행 (lazy import 유지)
        from search import search_image

        results = search_image(path, top_k=5)

        if not results:
            # 결과가 비어있는 경우는 거의 없지만 방어
            return {"top1": None, "top5": []}

        return {"top1": results[0], "top5": results}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    finally:
        # ✅ 임시 파일 삭제 (무조건 정리)
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
