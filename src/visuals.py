import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import HubertModel, Wav2Vec2FeatureExtractor


AUDIO_PATH = "../data/test_audio/new_audio.wav"

# ---------- LOAD AUDIO ----------
y, sr = librosa.load(AUDIO_PATH, sr=None)

# ---------- IF NOT 16k → RESAMPLE ----------
TARGET_SR = 16000
if sr != TARGET_SR:
    y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    sr = TARGET_SR
    print(f"Resampled audio to {sr} Hz (from original sample rate).")

# ---------- WAVEFORM ----------
plt.figure(figsize=(12, 4))
plt.plot(np.linspace(0, len(y)/sr, len(y)), y)
plt.title("Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

# ---------- MFCC ----------
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc, x_axis="time")
plt.colorbar(format="%+2.f")
plt.title("MFCC Heatmap")
plt.tight_layout()
plt.show()

# ---------- HUBERT EMBEDDINGS ----------
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
hubert.eval()

inputs = feature_extractor(
    y,
    sampling_rate=sr,
    return_tensors="pt",
    padding=True
)

with torch.no_grad():
    outputs = hubert(inputs.input_values)
emb = outputs.last_hidden_state.squeeze().numpy()

plt.figure(figsize=(12, 4))
plt.imshow(emb.T, aspect="auto", origin="lower")
plt.colorbar()
plt.title("HuBERT Embedding Heatmap")
plt.tight_layout()
plt.show()
