import os
import torch
import librosa
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ======================
# CONFIG
# ======================
DATASET_PATH = r"H:\Ai_Project\audio_kill_detector\kill_detector\dataset\samples"
BATCH_SIZE = 16
EPOCHS = 50
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# FEATURE
# ======================
def extract_mel(y, sr=22050, duration=0.5):
    target_len = int(sr * duration)

    # ✅ force exact audio length
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=64,
        hop_length=512,   # 🔥 FIXED
        n_fft=2048        # 🔥 FIXED
    )

    mel = librosa.power_to_db(mel)

    # ✅ normalize
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)

    return mel.astype(np.float32)

# ======================
# DATASET
# ======================
class AudioDataset(Dataset):
    def __init__(self, root):
        self.files = []
        self.labels = []

        for label, folder in enumerate(["non_kill", "kill"]):
            path = os.path.join(root, folder)
            for f in os.listdir(path):
                if f.endswith((".wav", ".mp3")):
                    self.files.append(os.path.join(path, f))
                    self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        y, sr = librosa.load(self.files[idx], sr=22050)

        mel = extract_mel(y, sr)

        mel = np.expand_dims(mel, axis=0)  # (1, H, W)

        return torch.tensor(mel), torch.tensor(self.labels[idx], dtype=torch.float32)

# ======================
# MODEL
# ======================
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)

        # 👇 placeholder (we compute dynamically)
        self.fc1 = None
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        x = torch.flatten(x, 1)

        # 🔥 Initialize FC dynamically
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.shape[1], 64).to(x.device)

        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x

# ======================
# TRAIN
# ======================
dataset = AudioDataset(DATASET_PATH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = CNN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCELoss()
model.eval()
val_correct = 0
val_total = 0

correct = 0
total = 0

for epoch in range(EPOCHS):
    total_loss = 0

    for X, y in loader:
        X = X.to(DEVICE)
        y = y.to(DEVICE).unsqueeze(1)

        preds = model(X)
        loss = criterion(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        
        preds_binary = (preds > 0.5).float()
        correct += (preds_binary == y).sum().item()
        total += y.size(0)
        val_correct += (preds_binary == y).sum().item()
        val_total += y.size(0)

    total_loss /= len(dataset)
    val_acc = val_correct / val_total
    accuracy = correct / total
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Train Acc: {accuracy:.4f}, Val Acc: {val_acc:.4f}")

# SAVE MODEL
torch.save(model.state_dict(), "kill_cnn.pth")
print("Model saved: kill_cnn.pth")