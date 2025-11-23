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

# ========= Load HuBERT ================
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
hubert.eval()

# ========= YOUR REAL AUDIO DIRECTORY ============
AUDIO_DIR = r"C:\Users\amish\OneDrive\Desktop\speech-accent-project\data\IndicAccentDB"
LABELS = ["andhra_pradesh", "gujrat", "jharkhand", "karnataka", "kerala", "tamil"]

# ========= Load audio file list ============
audio_files = []
for label in LABELS:
    dir_path = os.path.join(AUDIO_DIR, label)
    if not os.path.isdir(dir_path):
        print(f"Skipping missing folder: {dir_path}")
        continue

    for f in os.listdir(dir_path):
        if f.endswith(".wav"):
            audio_files.append((os.path.join(dir_path, f), LABELS.index(label)))

print("Loaded:", len(audio_files), "audio files")


# ========= SAFE FEATURE EXTRACTION (NO CRASHES) ============
def extract_layer_features(path, layer):
    speech, sr = sf.read(path)

    # --- SKIP EMPTY OR BROKEN AUDIO FILES ---
    if speech is None or len(speech) < 200:
        return None

    # Convert to torch
    speech = torch.tensor(speech, dtype=torch.float32)

    # --- PRE-PAD (fix ultra-short audio BEFORE resampling) ---
    if speech.shape[0] < 5000:
        speech = F.pad(speech, (0, 5000 - speech.shape[0]))

    # --- RESAMPLE TO 16K ---
    if sr != 16000:
        try:
            speech = torchaudio.functional.resample(speech, sr, 16000)
        except:
            return None  # skip if resampling fails

    # --- POST-PAD (make sure HuBERT won't crash) ---
    if speech.shape[0] < 8000:
        speech = F.pad(speech, (0, 8000 - speech.shape[0]))

    # Convert for HuBERT
    speech = speech.numpy()

    inputs = feature_extractor(speech, sampling_rate=16000, return_tensors="pt")

    try:
        with torch.no_grad():
            outputs = hubert(**inputs, output_hidden_states=True)
    except:
        return None  # skip if ANY HuBERT error occurs

    hidden = outputs.hidden_states[layer]
    return hidden.mean(dim=1).squeeze().numpy()


# ========= LAYER-WISE EVALUATION ============
layer_accuracies = []

for layer in range(13):
    print(f"\nEvaluating Layer {layer} ...")

    X = []
    y = []

    for path, label in audio_files:
        feat = extract_layer_features(path, layer)
        if feat is None:
            continue  # skip broken audio

        X.append(feat)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        print("No valid files for this layer.")
        continue

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=5000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Layer {layer} Accuracy = {acc:.4f}")
    layer_accuracies.append((layer, acc))


# ========= PRINT RESULTS ============
print("\n=== FINAL LAYER-WISE RESULTS ===")
for layer, acc in layer_accuracies:
    print(f"Layer {layer}: {acc:.4f}")
