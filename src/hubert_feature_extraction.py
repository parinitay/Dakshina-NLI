import os
import torch
import librosa
import numpy as np
from transformers import HubertModel, Wav2Vec2FeatureExtractor

# ---------- Paths ----------
DATA_DIR = r"C:\Users\amish\OneDrive\Desktop\speech-accent-project\data\IndicAccentDB"
OUTPUT_DIR = r"C:\Users\amish\OneDrive\Desktop\speech-accent-project\features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Load HuBERT ----------
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
model.eval()
print("✅ HuBERT model loaded successfully!")

# ---------- Function ----------
def extract_hubert_features(file_path):
    wav, sr = librosa.load(file_path, sr=16000)
    inputs = feature_extractor(wav, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # mean-pool all frame embeddings → one 768-dim vector
    embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy()
    return embedding

# ---------- Loop through every folder ----------
for state in os.listdir(DATA_DIR):
    state_path = os.path.join(DATA_DIR, state)
    if not os.path.isdir(state_path):
        continue
    print(f"\n📁 Processing state: {state}")
    state_embeddings = []
    for fname in os.listdir(state_path):
        if fname.endswith(".wav"):
            fpath = os.path.join(state_path, fname)
            emb = extract_hubert_features(fpath)
            state_embeddings.append(emb)
    if state_embeddings:
        arr = np.vstack(state_embeddings)
        np.save(os.path.join(OUTPUT_DIR, f"{state}_features.npy"), arr)
        print(f"✅ Saved {state} → shape {arr.shape}")

