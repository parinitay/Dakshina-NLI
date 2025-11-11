# src/extract_all_features.py
import os
import numpy as np
from tqdm import tqdm
from model_hubert import get_hubert_embeddings_from_file

# --- Paths ---
BASE_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/IndicAccentDB"))
FEATURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/features"))

# Make sure features_dir exists
os.makedirs(FEATURES_DIR, exist_ok=True)

# --- Loop through each state folder ---
for state in os.listdir(BASE_DATA_DIR):
    state_path = os.path.join(BASE_DATA_DIR, state)
    if not os.path.isdir(state_path):
        continue

    print(f"\n🎧 Processing state: {state}")
    state_feature_dir = os.path.join(FEATURES_DIR, state)
    os.makedirs(state_feature_dir, exist_ok=True)

    wav_files = [f for f in os.listdir(state_path) if f.lower().endswith(".wav")]

    for file_name in tqdm(wav_files, desc=f"Processing {state}", ncols=100):
        file_path = os.path.join(state_path, file_name)
        output_path = os.path.join(state_feature_dir, file_name.replace(".wav", ".npy"))

        # Skip if already extracted
        if os.path.exists(output_path):
            continue

        # Extract embeddings
        try:
            emb = get_hubert_embeddings_from_file(file_path)
            if emb is not None:
                np.save(output_path, emb)
            else:
                print(f"❌ Failed to extract for {file_name}")
        except Exception as e:
            print(f"❌ Error processing {file_name}: {e}")

print("\n🏁 Feature extraction complete!")
