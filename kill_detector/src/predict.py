import pickle
import librosa
import numpy as np
import subprocess
import os
from kill_detector.src.onset_detector import detect_onsets

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

    # ✅ Load audio (FIXED)
    y, sr = librosa.load(wav_path)

    # Detect onsets
    onset_times = detect_onsets(wav_path)

    events = []

    for onset_time in onset_times:
        start_sample = int(onset_time * sr)
        end_sample = int((onset_time + 0.5) * sr)

        # ✅ Bounds check
        if end_sample > len(y):
            continue

        frame = y[start_sample:end_sample]

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
    if y is None or len(y) == 0:
        return None

    if len(y) < sr * 0.1:
        return None

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    if mfccs.size == 0:
        return None

    features = np.concatenate([
        np.mean(mfccs, axis=1), np.std(mfccs, axis=1),
        np.mean(spectral_centroid, axis=1), np.std(spectral_centroid, axis=1),
        np.mean(zcr, axis=1), np.std(zcr, axis=1),
        np.mean(rolloff, axis=1), np.std(rolloff, axis=1)
    ])

    if np.any(np.isnan(features)):
        return None

    return features

def extract_wav_from_video(video_path):
    output_path = video_path.replace(".mp4", ".wav")

    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-ac", "1",
        "-ar", "22050",
        output_path
    ]

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return output_path