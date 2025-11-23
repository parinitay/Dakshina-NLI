import numpy as np
import joblib
import torch
import soundfile as sf
import torchaudio
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import os

# -----------------------------------
# 1. CORRECT MODEL PATHS FOR YOUR PC
# -----------------------------------
MODEL_PATH = "./models/accent_classifier.pkl"
ENCODER_PATH = "./models/label_encoder.pkl"

clf = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

# -----------------------------------
# 2. LOAD HUBERT
# -----------------------------------
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
hubert.eval()

# -----------------------------------
# 3. CHILD AUDIO FOLDER
# -----------------------------------
# Put child1_male.wav and child2_female.wav here:
# speech-accent-project/data/child_test/
CHILD_DIR = r"../data/child_test"

# -----------------------------------
# 4. FEATURE EXTRACTION FOR 1 AUDIO
# -----------------------------------
def extract_features(path):
    speech, sr = sf.read(path)

    # Convert stereo → mono if needed
    if len(speech.shape) > 1:
        speech = np.mean(speech, axis=1)

    # Resample to 16kHz
    if sr != 16000:
        speech = torchaudio.functional.resample(
            torch.tensor(speech, dtype=torch.float32),
            sr,
            16000
        ).numpy()

    inputs = feature_extractor(speech, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        outputs = hubert(**inputs, output_hidden_states=True)

    # Use Layer 2 (highest accuracy from your layer analysis)
    hidden = outputs.hidden_states[2]

    return hidden.mean(dim=1).squeeze().numpy()

# -----------------------------------
# 5. RUN PREDICTIONS FOR CHILD AUDIO
# -----------------------------------
print("🎯 Testing on Child Audio Samples...\n")

files = [f for f in os.listdir(CHILD_DIR) if f.endswith(".wav")]

for f in files:
    path = os.path.join(CHILD_DIR, f)
    feat = extract_features(path).reshape(1, -1)

    pred = clf.predict(feat)[0]
    pred_label = encoder.inverse_transform([pred])[0]

    print(f"File: {f}")
    print(f"Predicted Accent: {pred_label}")
    print("--------------------")
