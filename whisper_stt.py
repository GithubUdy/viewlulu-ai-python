import whisper
import os
import tempfile

# 서버 시작 시 1회 로딩
model = whisper.load_model("base")  # tiny / base / small 가능

def transcribe_audio(file_bytes: bytes, filename: str):
    suffix = os.path.splitext(filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="tmp") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
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
        os.remove(tmp_path)
