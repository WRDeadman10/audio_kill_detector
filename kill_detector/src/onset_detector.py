import librosa
import numpy as np
import logging

logger = logging.getLogger(__name__)

def detect_onsets(audio_path):
    try:
        logger.info(f"Processing audio file: {audio_path}")
        y, sr = librosa.load(audio_path)
        if len(y) < 100:  # Assuming audio shorter than 100 samples is too short
            logger.warning("Audio file is too short to detect onsets.")
            return []

        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        logger.info(f"Detected {len(onset_times)} onsets.")
        return sorted(onset_times)
    except Exception as e:
        logger.error(f"Error detecting onsets: {e}")
        return []
