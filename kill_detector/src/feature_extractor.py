import librosa
import numpy as np

def extract_features(audio_path: str) -> np.ndarray:
    try:
        y, sr = librosa.load(audio_path)

        # 🚫 Skip empty audio
        if y is None or len(y) == 0:
            return None # type: ignore

        # 🚫 Skip too short audio
        if len(y) < sr * 0.1:  # < 100ms
            return None # type: ignore


        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

        # 🚫 Validate shapes
        if mfccs.size == 0 or spectral_centroid.size == 0:
            return None # type: ignore


        feature_vector = np.concatenate([
            np.mean(mfccs, axis=1), np.std(mfccs, axis=1),
            np.mean(spectral_centroid, axis=1), np.std(spectral_centroid, axis=1),
            np.mean(zero_crossing_rate, axis=1), np.std(zero_crossing_rate, axis=1),
            np.mean(spectral_rolloff, axis=1), np.std(spectral_rolloff, axis=1)
        ])

        # 🚫 Handle NaNs
        if np.any(np.isnan(feature_vector)):
            return None # type: ignore


        return feature_vector

    except Exception as e:
        print(f"[ERROR] {audio_path}: {e}")
        return None # type: ignore
