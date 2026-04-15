import pickle
import librosa
import numpy as np
import subprocess
import os
from onset_detector import detect_onsets
from audio_extractor import extract_wav_from_video
import utils

def predict_events(
    video_path,
    model_path="models/kill_model.pkl",
    confidence_threshold=0.5,
    min_gap_seconds=0.3
):
    # Load model
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)

    # ✅ Extract WAV
    wav_path = extract_wav_from_video(video_path)

    if wav_path is None or not os.path.exists(wav_path):
        raise RuntimeError("Audio extraction failed")

    # ✅ Load audio (FIXED)
    y, sr = librosa.load(wav_path)

    # Detect onsets
    onset_times = detect_onsets(wav_path)
    print(f"Loaded {len(onset_times)} onset candidates")
    events = []

    for onset_time in onset_times:
        print(f"Processing onset at {onset_time:.2f}s")
        start_sample = int(onset_time * sr)
        end_sample = int((onset_time + 0.5) * sr)

        # ✅ Bounds check
        if end_sample > len(y):
            continue

        frame = y[start_sample:end_sample]
        
           # 🔥 ADD THIS BLOCK RIGHT HERE
        rms = librosa.feature.rms(y=frame)[0][0]

        if rms < 0.02:  # tune this
            continue

        # ✅ Extract features from array (see fix below)
        features = extract_features_from_array(frame, sr)

        if features is None:
            continue

        # sklearn expects 2D
        features = features.reshape(1, -1)

        prob = clf.predict_proba(features)[0][1]

        if prob >= confidence_threshold:
            events.append((onset_time, onset_time + 0.5, prob))

    # ✅ Gap filtering
    filtered_events = []
    last_end_time = -min_gap_seconds

    for start_time, end_time, prob in events:
        if start_time > last_end_time + min_gap_seconds:
            filtered_events.append((start_time, end_time, prob))
            last_end_time = end_time

    return filtered_events

def extract_features_from_array(y, sr):
    if len(y) == 0:
        return None

    # EXISTING FEATURES
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # 🔥 NEW FEATURES (critical)
    rms = librosa.feature.rms(y=y)
    
    peak_amplitude = np.max(np.abs(y))
    energy = np.sum(y**2)

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    features = np.concatenate([
        np.mean(mfccs, axis=1), np.std(mfccs, axis=1),

        # 🔥 ADD THESE
        [np.mean(rms), np.std(rms)],
        [peak_amplitude],
        [energy],

        np.mean(spectral_centroid, axis=1),
        np.std(spectral_centroid, axis=1),
        np.mean(rolloff, axis=1),
        np.std(rolloff, axis=1),
    ])

    return features

def prepare_input(mel_spec):
    # Normalize
    mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)

    # Add channel dimension
    return mel_spec[np.newaxis, :, :]  # (1, H, W)

if __name__ == "__main__":
    utils.setup_logger("audio_kill")
    video = r"H:\Ai_Project\audio_kill_detector\inputFolder\inputVideo.mp4"

    print("Starting prediction...")

    events = predict_events(video, confidence_threshold=0.2)

    print(f"\nTotal events detected: {len(events)}")

    for start, end, prob in events:
        print(f"[{start:.2f}s - {end:.2f}s] Confidence: {prob:.2f}")