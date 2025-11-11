# src/features_mfcc.py
import librosa
import numpy as np
import os
import pandas as pd
import pickle

def extract_mfcc(audio_path, n_mfcc=13, duration=3, sr=16000):
    """Extract MFCC features from an audio file."""
    try:
        y, sr = librosa.load(audio_path, sr=sr, duration=duration)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        return mfcc_mean
    except Exception as e:
        print(f"⚠️ Error processing {audio_path}: {e}")
        return None


def process_dataset(metadata_csv="data/IndicAccentDB/metadata.csv", out_file="data/features.pkl"):
    """Load metadata, extract MFCCs for each audio file, and save features."""
    print("🎧 Extracting MFCC features from dataset...")

    df = pd.read_csv(metadata_csv)
    features = []
    labels = []

    for idx, row in df.iterrows():
        audio_path = row["path"]
        label = row["label"]

        if not os.path.isfile(audio_path):
            print(f"⚠️ Missing file: {audio_path}")
            continue

        mfcc = extract_mfcc(audio_path)
        if mfcc is not None:
            features.append(mfcc)
            labels.append(label)

    print(f"✅ Extracted features from {len(features)} audio samples")

    # Save features and labels as pickle
    os.makedirs("data", exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump({"features": np.array(features), "labels": np.array(labels)}, f)

    print(f"💾 Saved extracted features to {out_file}")


if __name__ == "__main__":
    process_dataset()
