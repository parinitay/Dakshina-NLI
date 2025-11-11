# src/predict_accent.py

import os
import numpy as np
import torch
import joblib
from model_hubert import get_hubert_embeddings_from_file

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "accent_classifier.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")

# Load model and encoder
clf = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

def predict_accent(audio_path):
    print(f"🎧 Processing: {audio_path}")
    emb = get_hubert_embeddings_from_file(audio_path)

    if emb is None:
        print("❌ Could not extract features.")
        return

    # Mean-pool to 1D if needed
    if emb.ndim > 1:
        emb = emb.mean(axis=0)

    # Predict
    emb = emb.reshape(1, -1)
    pred = clf.predict(emb)
    accent = label_encoder.inverse_transform(pred)[0]
    print(f"✅ Predicted Accent: {accent}")

# Run the function
if __name__ == "__main__":
    test_audio = os.path.join(BASE_DIR, "../data/test_audio/new_audio.wav")
    predict_accent(test_audio)
