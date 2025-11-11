# src/model_hubert.py
import torch
import torchaudio

# Load HuBERT model once (adjust as needed)
bundle = torchaudio.pipelines.HUBERT_BASE
model = bundle.get_model()

def get_hubert_embeddings_from_file(file_path):
    """
    Extract HuBERT embeddings from a given audio file.
    Returns a numpy array of embeddings.
    """
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Resample if needed
        if sample_rate != bundle.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=bundle.sample_rate)
            waveform = resampler(waveform)
        
        with torch.inference_mode():
            features, _ = model(waveform)
        
        # Convert to numpy
        embeddings = features.squeeze(0).numpy()
        return embeddings
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None



