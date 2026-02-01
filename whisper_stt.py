import whisper
import os
import tempfile
import subprocess

# ==================================================
# Whisper STT Core Logic (FINAL STABLE)
# ==================================================

# 🔥 서버 시작 시 1회만 로딩
model = whisper.load_model("base")

def transcribe_audio(file_bytes: bytes, filename: str):
    """
    📌 Whisper 음성 인식 처리
    - bytes → 원본 임시 파일(mp4/aac 가능)
    - ffmpeg → wav 변환
    - Whisper는 wav만 처리
    """

    os.makedirs("tmp", exist_ok=True)

    # 1️⃣ 원본 파일 확장자 유지
    orig_suffix = os.path.splitext(filename)[1] or ".bin"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=orig_suffix,
        dir="tmp"
    ) as src:
        src.write(file_bytes)
        src.flush()
        os.fsync(src.fileno())
        src_path = src.name

    # 2️⃣ wav 변환 경로
    wav_path = src_path + ".wav"

    try:
        # 3️⃣ ffmpeg 변환 (🔥 핵심)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", src_path,
                "-ac", "1",
                "-ar", "16000",
                wav_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        # 4️⃣ Whisper 실행
        result = model.transcribe(
            wav_path,
            language="ko",
            fp16=False,
            temperature=0.0,
            no_speech_threshold=0.3
        )

        text = result.get("text", "").strip()

        return {
            "text": text,
            "contains_chalkak": any(
                kw in text
                for kw in ["찰칵", "김치", "치즈", "브이", "사진", "촬영", "찰칵찰칵"]
            ),
        }

    finally:
        # 5️⃣ 임시 파일 정리
        for p in (src_path, wav_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
