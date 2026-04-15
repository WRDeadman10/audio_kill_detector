import torch
import torch.nn as nn
import librosa
import numpy as np
import subprocess
import os

# ======================
# CONFIG
# ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WINDOW_SECONDS = 0.5
HOP_SECONDS = 0.1
THRESHOLD = 0.7

# ======================
# MODEL
# ======================
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(2560, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        x = torch.flatten(x, 1)

        if self.fc1 is None:
            self.fc1 = nn.Linear(x.shape[1], 64).to(x.device)

        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x

# ======================
# LOAD MODEL
# ======================
model = CNN().to(DEVICE)
model.load_state_dict(torch.load("kill_cnn.pth", map_location=DEVICE))
model.eval()

# ======================
# AUDIO UTILS
# ======================
import uuid

def extract_wav(video_path):
    output = f"temp_{uuid.uuid4().hex}.wav"  # 🔥 unique file every time

    result = subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1", "-ar", "22050",
        output
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print("FFmpeg Error:", result.stderr)
        raise RuntimeError("Audio extraction failed")

    if not os.path.exists(output):
        raise RuntimeError("WAV not created")

    print("Generated:", output)  # 🔥 debug

    return output

def extract_mel(y, sr=22050):
    target_len = int(sr * WINDOW_SECONDS)

    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=64,
        hop_length=512,
        n_fft=2048
    )

    mel = librosa.power_to_db(mel)

    mel = (mel - mel.mean()) / (mel.std() + 1e-6)

    return mel.astype(np.float32)

# ======================
# CORE DETECTION
# ======================
def predict(video_path):
    wav = extract_wav(video_path)
    y, sr = librosa.load(wav, sr=22050)

    window = int(sr * WINDOW_SECONDS)
    hop = int(sr * HOP_SECONDS)

    events = []

    for i in range(0, len(y) - window, hop):
        frame = y[i:i+window]

        mel = extract_mel(frame, sr)
        mel = np.expand_dims(mel, axis=(0,1))

        tensor = torch.tensor(mel).to(DEVICE)

        with torch.no_grad():
            prob = model(tensor).item()

        # soften extreme confidence (optional but useful)
        prob = prob ** 2

        if prob > THRESHOLD:
            time = i / sr
            events.append((time, prob))

    return events

# ======================
# CLUSTER EVENTS
# ======================
def cluster_events(events, time_gap=0.4, min_count=3):
    if not events:
        return []

    clusters = []
    current = [events[0]]

    for e in events[1:]:
        if e[0] - current[-1][0] <= time_gap:
            current.append(e)
        else:
            if len(current) >= min_count:
                clusters.append(current)
            current = [e]

    if len(current) >= min_count:
        clusters.append(current)

    return clusters

# ======================
# PICK PEAKS
# ======================
def extract_peaks(clusters):
    final = []

    for cluster in clusters:
        best = max(cluster, key=lambda x: x[1])
        final.append(best)

    return final

# ======================
# RUN
# ======================
if __name__ == "__main__":
    video = r"H:\Ai_Project\audio_kill_detector\inputFolder\inputVideo2.mp4"

    print("Running CNN detection...")

    raw_events = predict(video)

    print(f"\nRaw detections: {len(raw_events)}")

    # clustering
    clusters = cluster_events(raw_events)

    # peak selection
    final_events = extract_peaks(clusters)

    print("\nFinal Kill Events:")
    for t, p in final_events:
        print(f"{t:.2f}s → {p:.2f}")