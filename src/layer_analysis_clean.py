import torch
import numpy as np
import torchaudio
import torch.nn.functional as F
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import soundfile as sf

# Load HuBERT
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
hubert.eval()

# Load clean list
clean_files = []
with open("clean_audio_list.txt", "r") as f:
    for line in f:
        path, label = line.strip().split("|")
        clean_files.append((path, int(label)))

print("Using clean files:", len(clean_files))

# Safe feature extraction
def extract_layer_features(path, layer):
    data, sr = sf.read(path)
    data = torch.tensor(data, dtype=torch.float32)

    # Pad minimum length
    if len(data) < 5000:
        data = F.pad(data, (0, 5000 - len(data)))

    # Resample
    if sr != 16000:
        data = torchaudio.functional.resample(data, sr, 16000)

    # After resample pad again
    if len(data) < 8000:
        data = F.pad(data, (0, 8000 - len(data)))

    data = data.numpy()

    inputs = feature_extractor(data, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        outputs = hubert(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states[layer]
    return hidden.mean(dim=1).squeeze().numpy()

# Layer analysis
results = []

for layer in range(13):
    print(f"Evaluating Layer {layer} ...")

    X = []
    y = []

    for path, label in clean_files:
        feat = extract_layer_features(path, layer)
        X.append(feat)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = LogisticRegression(max_iter=5000)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    acc = accuracy_score(y_test, pred)
    print(f"Layer {layer} Accuracy = {acc:.4f}")
    results.append((layer, acc))

print("\n=== CLEAN LAYER RESULTS ===")
for layer, acc in results:
    print(layer, acc)
