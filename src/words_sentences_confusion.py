import os
import joblib
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from transformers import Wav2Vec2FeatureExtractor, HubertModel
import torch
import torchaudio

# -------------------------
# FIXED PATHS
# -------------------------
MODEL_PATH = r"C:\Users\amish\OneDrive\Desktop\speech-accent-project\src\models\accent_classifier.pkl"
ENCODER_PATH = r"C:\Users\amish\OneDrive\Desktop\speech-accent-project\src\models\label_encoder.pkl"

WORD_DIR = r"C:\Users\amish\OneDrive\Desktop\speech-accent-project\data\word_samples\andhra_pradesh"
SENT_DIR = r"C:\Users\amish\OneDrive\Desktop\speech-accent-project\data\sentence_samples\andhra_pradesh"

clf = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
hubert.eval()

TRUE_LABEL = encoder.transform(["andhra_pradesh"])[0]

# -------------------------
# FIX: PAD SHORT AUDIO
# -------------------------
def pad_audio(audio, target_len=16000):
    if len(audio) < target_len:
        pad_width = target_len - len(audio)
        audio = np.pad(audio, (0, pad_width))
    return audio


# -------------------------
# Extract Features
# -------------------------
def extract_features(path):
    speech, sr = sf.read(path)

    # convert stereo → mono
    if speech.ndim == 2:
        speech = speech.mean(axis=1)

    # resample to 16k
    if sr != 16000:
        speech = torchaudio.functional.resample(
            torch.tensor(speech).float(), sr, 16000
        ).numpy()

    # PAD SHORT clips
    speech = pad_audio(speech, 16000)

    inputs = feature_extractor(speech, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        outputs = hubert(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states[12]
    return hidden.mean(dim=1).squeeze().numpy()


def load_files(folder):
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".wav")
    ]


word_files = load_files(WORD_DIR)
sent_files = load_files(SENT_DIR)

print("Word files:", len(word_files))
print("Sentence files:", len(sent_files))

# -------------------------
# Predictions
# -------------------------
y_true = []
y_pred = []

print("\nExtracting WORD predictions...")
for file in word_files:
    feat = extract_features(file).reshape(1, -1)
    pred = clf.predict(feat)[0]
    y_true.append(TRUE_LABEL)
    y_pred.append(pred)

print("\nExtracting SENTENCE predictions...")
for file in sent_files:
    feat = extract_features(file).reshape(1, -1)
    pred = clf.predict(feat)[0]
    y_true.append(TRUE_LABEL)
    y_pred.append(pred)

# -------------------------
# Confusion Matrix
# -------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_,
    cmap="Purples"
)
plt.title("Confusion Matrix: Word vs Sentence Level")
plt.xlabel("Predicted Accent")
plt.ylabel("True Accent")
plt.tight_layout()
plt.show()

print("\nCONFUSION MATRIX:\n", cm)


