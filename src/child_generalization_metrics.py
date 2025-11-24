import numpy as np
import joblib
import torch
import soundfile as sf
import torchaudio
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# -------------------------------
# 1. Paths (YOUR MODEL PATHS)
# -------------------------------
MODEL_PATH = "./models/accent_classifier.pkl"
ENCODER_PATH = "./models/label_encoder.pkl"

clf = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

# -------------------------------
# 2. Load HuBERT
# -------------------------------
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
hubert.eval()

# -------------------------------
# 3. Child audio folder
# -------------------------------
CHILD_DIR = r"../data/child_test1"


# -------------------------------
# 4. Feature extraction
# -------------------------------
def extract_features(path):
    speech, sr = sf.read(path)

    # stereo → mono
    if len(speech.shape) > 1:
        speech = np.mean(speech, axis=1)

    # resample
    if sr != 16000:
        speech = torchaudio.functional.resample(
            torch.tensor(speech, dtype=torch.float32), sr, 16000
        ).numpy()

    inputs = feature_extractor(speech, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        outputs = hubert(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states[2]   # best layer
    return hidden.mean(dim=1).squeeze().numpy()


# -------------------------------
# 5. Load & predict on child data
# -------------------------------
true_labels = []
pred_labels = []

print("\n🎯 Testing on Child Audio + Generating Metrics...\n")

for accent_folder in os.listdir(CHILD_DIR):
    folder_path = os.path.join(CHILD_DIR, accent_folder)

    if not os.path.isdir(folder_path):
        continue

    # true label is the folder name
    true_label_index = encoder.transform([accent_folder])[0]

    for f in os.listdir(folder_path):
        if f.endswith(".wav"):
            filepath = os.path.join(folder_path, f)

            feat = extract_features(filepath).reshape(1, -1)
            prediction = clf.predict(feat)[0]

            true_labels.append(true_label_index)
            pred_labels.append(prediction)

            predicted_accent = encoder.inverse_transform([prediction])[0]

            print(f"File: {f}")
            print(f"True Accent: {accent_folder}")
            print(f"Predicted Accent: {predicted_accent}")
            print("----------------------------------")


# -------------------------------
# 6. Metrics
# -------------------------------
acc = accuracy_score(true_labels, pred_labels)
prec = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
rec = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
cm = confusion_matrix(true_labels, pred_labels)

print("\n===============================")
print("📊 CHILD GENERALIZATION METRICS")
print("===============================")
print(f"Accuracy: {acc:.4f}")
print(f"Precision (macro): {prec:.4f}")
print(f"Recall (macro): {rec:.4f}")
print(f"F1 Score (macro): {f1:.4f}")
print("\nConfusion Matrix:\n", cm)


# -------------------------------
# 7. Plot confusion matrix
# -------------------------------
labels = encoder.classes_

plt.figure(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Greens", values_format='d')
plt.title("Child Audio Accent Confusion Matrix")
plt.tight_layout()
plt.savefig("../child_confusion_matrix.png")  # saved in project root
plt.show()
