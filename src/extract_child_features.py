import os
import numpy as np
import torch                      # <<< FIX ADDED HERE
import soundfile as sf
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, HubertModel

# Paths
CHILD_DIR = "../data/child_test"
SAVE_FEATURES = "../data/features/child_features.npy"
SAVE_LABELS = "../data/features/child_labels.npy"

# Load HuBERT
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
hubert.eval()

def extract_feature(path):
    audio, sr = sf.read(path)

    # Resample
    if sr != 16000:
        audio = torchaudio.functional.resample(
            torch.tensor(audio, dtype=torch.float32), sr, 16000
        ).numpy()

    inp = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        out = hubert(**inp, output_hidden_states=True)

    # Final layer features
    return out.hidden_states[-1].mean(dim=1).squeeze().numpy()


features = []
labels = []

for label in os.listdir(CHILD_DIR):
    folder = os.path.join(CHILD_DIR, label)
    if not os.path.isdir(folder):
        continue

    for file in os.listdir(folder):
        if file.endswith(".wav"):
            path = os.path.join(folder, file)
            print("Extracting:", path)
            feat = extract_feature(path)
            features.append(feat)
            labels.append(label)

np.save(SAVE_FEATURES, np.array(features))
np.save(SAVE_LABELS, np.array(labels))

print("Saved:", SAVE_FEATURES, SAVE_LABELS)
