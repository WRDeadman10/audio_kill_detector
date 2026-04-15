import subprocess
import os
import tempfile


def extract_wav_from_video(video_path: str) -> str:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # ✅ Always generate temp WAV (no overwrite issues)
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, "kill_detector_audio.wav")

    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",                 # 🚀 ignore video stream
        "-acodec", "pcm_s16le",# 🚀 force WAV format
        "-ar", "22050",        # sample rate
        "-ac", "1",            # mono
        output_path
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    # 🔴 If ffmpeg fails → STOP immediately
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{result.stderr}")

    # 🔴 Double-check file actually exists
    if not os.path.exists(output_path):
        raise RuntimeError("WAV file was not created")

    # 🔴 Check file size (common silent failure)
    if os.path.getsize(output_path) < 1000:
        raise RuntimeError("Generated WAV file is too small / invalid")

    return output_path