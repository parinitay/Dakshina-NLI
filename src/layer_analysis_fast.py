import torch
import numpy as np
import torchaudio
import torch.nn.functional as F
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import soundfile as sf
import os
import random

# ========= Load HuBERT ================
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
hubert.eval()

AUDIO_DIR = r"C:\Users\amish\OneDrive\Desktop\speech-accent-project\data\IndicAccentDB"
LABELS = ["andhra_pradesh", "gujrat", "jharkhand", "karnataka", "kerala", "tamil"]

MAX_FILES_PER_CLASS = 80   # << FAST MODE

# ========= Collect sample files ============
audio_files = []
for label in LABELS:
    dir_path = os.path.join(AUDIO_DIR, label)
    all_files = [f for f in os.listdir(dir_path) if f.endswith(".wav")]

    # Pick only 80 random files per class
    selected_files = random.sample(all_files, min(MAX_FILES_PER_CLASS, len(all_files)))

    for f in selected_files:
        audio_files.append((os.path.join(dir_path, f), LABELS.index(label)))

print("FAST MODE: Loaded", len(audio_files), "files")

# ========= Safe feature extraction ========
def extract_features_batch(paths, layer):
    batch = []

    for path in paths:
        speech, sr = sf.read(path)
        if speech is None or len(speech) < 200:
            continue

        speech = torch.tensor(speech, dtype=torch.float32)

        if speech.shape[0] < 5000:
            speech = F.pad(speech, (0, 5000 - speech.shape[0]))

        if sr != 16000:
            try:
                speech = torchaudio.functional.resample(speech, sr, 16000)
            except:
                continue

        if speech.shape[0] < 8000:
            speech = F.pad(speech, (0, 8000 - speech.shape[0]))

        batch.append(speech.numpy())

    if len(batch) == 0:
        return None

    inputs = feature_extractor(batch, sampling_rate=16000, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = hubert(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states[layer]
    hidden = hidden.mean(dim=1).numpy()

    return hidden


# ========= Run analysis fast ============
layer_accuracies = []
BATCH_SIZE = 16

for layer in range(13):
    print(f"\nEvaluating Layer {layer} ...")

    X = []
    y = []

    for label_idx, label in enumerate(LABELS):
        class_files = [p for p, l in audio_files if l == label_idx]

        for i in range(0, len(class_files), BATCH_SIZE):
            batch_paths = class_files[i:i+BATCH_SIZE]
            feats = extract_features_batch(batch_paths, layer)
            if feats is None:
                continue

            for f in feats:
                X.append(f)
                y.append(label_idx)

    if len(X) == 0:
        print("No valid data for this layer.")
        continue

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=3000)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)

    print(f"Layer {layer} Accuracy = {acc:.4f}")
    layer_accuracies.append((layer, acc))

print("\n=== FINAL FAST RESULTS ===")
for layer, acc in layer_accuracies:
    print(f"Layer {layer}: {acc:.4f}")
