import librosa
import numpy as np

def extract_features(audio_path: str, window_seconds: float = 0.5) -> np.ndarray:
    "Extract features for a given audio file"
    
    y, sr = librosa.load(audio_path)
    n_samples = int(sr * window_seconds)
    hop_length = int(sr * 0.01)  # 10ms hop length
    
    features = []
    for i in range(0, len(y), hop_length):
        frame = y[i:i+n_samples]
        if len(frame) < n_samples:
            break
        
        mfccs = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=frame, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(frame)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=frame, sr=sr)
        
        mfcc_mean_std = np.mean(mfccs, axis=1), np.std(mfccs, axis=1)
        centroid_mean_std = np.mean(spectral_centroid, axis=1), np.std(spectral_centroid, axis=1)
        zcr_mean_std = np.mean(zero_crossing_rate, axis=1), np.std(zero_crossing_rate, axis=1)
        rolloff_mean_std = np.mean(spectral_rolloff, axis=1), np.std(spectral_rolloff, axis=1)
        
        feature_vector = np.concatenate([mfcc_mean_std[0], mfcc_mean_std[1],
                                        centroid_mean_std[0], centroid_mean_std[1],
                                        zcr_mean_std[0], zcr_mean_std[1],
                                        rolloff_mean_std[0], rolloff_mean_std[1]])
        
        features.append(feature_vector)
    
    return np.array(features)
