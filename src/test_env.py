import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, HubertModel

print("✅ Torch version:", torch.__version__)
print("✅ Torchaudio version:", torchaudio.__version__)

# Try loading the HuBERT model
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960")

print("✅ HuBERT model loaded successfully!")
