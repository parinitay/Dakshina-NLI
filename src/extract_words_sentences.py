import os
import numpy as np
import soundfile as sf
import torchaudio
import torch
from transformers import HubertModel, Wav2Vec2FeatureExtractor

WORD_DIR = "../data/word_samples/andhra_pradesh"
SENT_DIR = "../data/sentence_samples/andhra_pradesh"

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
hubert.eval()

def extract_embedding(path, layer=2):
    speech, sr = sf.read(path)

    if len(speech.shape) > 1:
        speech = speech.mean(axis=1)

    if sr != 16000:
        speech = torchaudio.functional.resample(
            torch.tensor(speech, dtype=torch.float32),
            sr, 16000
        ).numpy()

    inputs = feature_extractor(speech, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        outputs = hubert(**inputs, output_hidden_states=True)

    return outputs.hidden_states[layer].mean(dim=1).squeeze().numpy()


def extract_from_folder(folder):
    X = []
    for f in os.listdir(folder):
        if f.endswith(".wav"):
            path = os.path.join(folder, f)
            emb = extract_embedding(path)
            X.append(emb)
    return np.array(X)


print("Extracting word-level features...")
X_words = extract_from_folder(WORD_DIR)
np.save("../data/word_X.npy", X_words)

print("Extracting sentence-level features...")
X_sent = extract_from_folder(SENT_DIR)
np.save("../data/sent_X.npy", X_sent)

print("Done!")
