import librosa
import numpy as np

def detect_onsets(audio_path):
    try:
        y, sr = librosa.load(audio_path)
        if len(y) < 100:  # Assuming audio shorter than 100 samples is too short
            return []

        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        return sorted(onset_times)
    except Exception as e:
        print(f"Error detecting onsets: {e}")
        return []
