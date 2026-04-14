import os
import numpy as np
from feature_extractor import extract_features
from onset_detector import detect_onsets

def predict_events(video_path, model_path="kill_detector/models/kill_model.pkl", confidence_threshold=0.5, min_gap_seconds=0.3):
    "Inference pipeline that loads model, extracts WAV from video, finds onset candidates, extracts 0.5s windows, computes features, predicts probabilities, keeps events above confidence threshold, and applies minimum 0.3s time-gap filtering."

    # Load the trained model
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)

    # Extract WAV from video (assuming a function extract_wav_from_video exists)
    wav_path = extract_wav_from_video(video_path)

    # Detect onsets
    onset_times = detect_onsets(wav_path)

    events = []
    for onset_time in onset_times:
        start_time = onset_time
        end_time = start_time + 0.5

        if end_time > len(y):
            break

        frame = y[int(start_time * sr):int(end_time * sr)]
        features = extract_features(frame)

        # Predict probability
        prob = clf.predict_proba(features)[0][1]

        if prob >= confidence_threshold:
            events.append((start_time, end_time, prob))

    # Apply minimum time-gap filtering
    filtered_events = []
    last_end_time = -min_gap_seconds

    for event in events:
        start_time, end_time, _ = event

        if start_time > last_end_time + min_gap_seconds:
            filtered_events.append(event)
            last_end_time = end_time

    return filtered_events
