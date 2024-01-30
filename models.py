import librosa
import numpy as np
import torch
from torch import nn


class AudioClassifier(nn.Module):
    def __init__(self, hidden_dim=128, feature_dim=128, device="cpu"):
        super(AudioClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Mish(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
        )
        self.device = device

    def forward(self, x):
        return self.fc(x)

    def infer_from_file(self, file_path):
        feature = extract_features(file_path)
        feature = (
            torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(self.device)
        )

        with torch.no_grad():
            pred = self.forward(feature)

        pred = pred.detach().cpu()
        return torch.sigmoid(pred).item()


def extract_features(file_path):
    return extract_features_logmel(file_path, n_mels=128)


def extract_features_logmel(file_path, n_mels=128):
    audio, sr = librosa.load(file_path, sr=None)
    audio = librosa.util.normalize(audio)
    melspec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    log_melspec = librosa.power_to_db(melspec)
    return np.mean(log_melspec, axis=1)
