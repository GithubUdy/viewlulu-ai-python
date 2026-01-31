import whisper
import os
import tempfile

# ==================================================
# Whisper STT Core Logic
# ==================================================

# 🔥 서버 시작 시 1회만 로딩 (필수)
# tiny / base / small 가능
model = whisper.load_model("tiny")

def transcribe_audio(file_bytes: bytes, filename: str):
    """
    📌 Whisper 음성 인식 처리
    - bytes → 임시 wav 파일 저장
    - Whisper는 반드시 '파일 경로'를 입력으로 받음
    """

    # 1️⃣ 확장자 보정 (filename 없거나 이상한 경우 대비)
    suffix = os.path.splitext(filename)[1]
    if not suffix:
        suffix = ".wav"

    # 2️⃣ tmp 디렉토리 보장 (pm2 / docker 환경 대비)
    os.makedirs("tmp", exist_ok=True)

    # 3️⃣ 임시 파일 생성
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
        dir="tmp"
    ) as tmp:
        tmp.write(file_bytes)
        tmp.flush()              # 🔥 중요: 디스크 flush
        tmp_path = tmp.name

    try:
        # 4️⃣ Whisper 실행 (경로 기반)
        result = model.transcribe(tmp_path, language="ko")
        text = result.get("text", "").strip()

        return {
            "text": text,
            "contains_chalkak": (
                "찰칵" in text or
                "김치" in text or
                "치즈" in text or
                "브이" in text or
                "사진" in text
            )
        }

    finally:
        # 5️⃣ 임시 파일 정리 (누락 방지)
        try:
            os.remove(tmp_path)
        except Exception:
            pass
