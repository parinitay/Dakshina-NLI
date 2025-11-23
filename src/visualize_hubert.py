import torch
import torchaudio
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import matplotlib.pyplot as plt

audio_path = "data/test_audio/new_audio.wav"

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
hubert.eval()

waveform, sr = torchaudio.load(audio_path)
waveform = waveform.mean(dim=0, keepdim=True)
waveform = torchaudio.functional.resample(waveform, sr, 16000)

inputs = feature_extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")

with torch.no_grad():
    outputs = hubert(inputs.input_values)
    emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

plt.figure(figsize=(14, 5))
plt.plot(emb)
plt.title("HuBERT Embedding (768-dim vector)")
plt.xlabel("Embedding Index")
plt.ylabel("Value")
plt.show()
