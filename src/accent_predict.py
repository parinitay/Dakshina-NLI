import torch
import numpy as np
import librosa
import joblib
from transformers import HubertModel, Wav2Vec2Processor

# ---------- Paths ----------
MODEL_PATH = "accent_classifier.pkl"
ENCODER_PATH = "label_encoder.pkl"

# Load trained classifier + encoder
classifier = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# Load HuBERT model
processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-base-ls960")
hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
hubert.eval()

def extract_hubert_features(file_path):
    wav, sr = librosa.load(file_path, sr=16000)
    inputs = processor(wav, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = hubert(**inputs)
    embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy()
    return embedding

def predict_accent(file_path):
    features = extract_hubert_features(file_path).reshape(1, -1)
    pred = classifier.predict(features)[0]
    accent = label_encoder.inverse_transform([pred])[0]
    print(f"🎤 Predicted Accent: {accent}")

# ---------- TEST ----------
if __name__ == "__main__":
    test_file = r"C:\Users\amish\OneDrive\Desktop\speech-accent-project\data\IndicAccentDB\kerala\Kerala_speaker_01_1_2.wav"
    predict_accent(test_file)
