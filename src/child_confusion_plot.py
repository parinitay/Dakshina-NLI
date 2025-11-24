import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import torch
import soundfile as sf
import torchaudio
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import os
from sklearn.metrics import confusion_matrix

# ------------------------------
# LOAD MODEL + ENCODER
# ------------------------------
MODEL_PATH = "./models/accent_classifier.pkl"
ENCODER_PATH = "./models/label_encoder.pkl"

clf = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

# ------------------------------
# LOAD HUBERT
# ------------------------------
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
hubert.eval()

# ------------------------------
# CHILD TEST FOLDER
# ------------------------------
CHILD_DIR = "../data/child_test1"

# ------------------------------
# Extract HuBERT Features
# ------------------------------
def extract_features(path):
    speech, sr = sf.read(path)

    if len(speech.shape) > 1:
        speech = np.mean(speech, axis=1)

    if sr != 16000:
        speech = torchaudio.functional.resample(
            torch.tensor(speech, dtype=torch.float32), sr, 16000
        ).numpy()

    inputs = feature_extractor(speech, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        outputs = hubert(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states[2]
    return hidden.mean(dim=1).squeeze().numpy()


# ------------------------------
# RUN PREDICTIONS
# ------------------------------
true_labels = []
pred_labels = []

print("\n🎯 Running Child Confusion Matrix Test...\n")

for accent_folder in os.listdir(CHILD_DIR):

    folder_path = os.path.join(CHILD_DIR, accent_folder)

    for file in os.listdir(folder_path):

        if file.endswith(".wav"):
            fpath = os.path.join(folder_path, file)

            feat = extract_features(fpath).reshape(1, -1)
            pred = clf.predict(feat)[0]

            true_labels.append(accent_folder)
            pred_labels.append(encoder.inverse_transform([pred])[0])

            print(f"File: {file}")
            print(f"True: {accent_folder} | Pred: {encoder.inverse_transform([pred])[0]}")
            print("-------------------------")

# ------------------------------
# CONFUSION MATRIX
# ------------------------------
labels_sorted = sorted(list(set(true_labels + pred_labels)))
cm = confusion_matrix(true_labels, pred_labels, labels=labels_sorted)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=labels_sorted,
    yticklabels=labels_sorted
)
plt.title("Child Accent Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("../data/child_test1/confusion_matrix_child.png")
plt.show()
