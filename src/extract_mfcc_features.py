import os
import numpy as np
import librosa

DATA_DIR = "../data/IndicAccentDB"
OUT_DIR = "../data/features/mfcc"

os.makedirs(OUT_DIR, exist_ok=True)

def extract_mfcc_fast(path):
    # Load audio fast in mono, 16k
    audio, sr = librosa.load(path, sr=16000, mono=True)

    # Compute 40 MFCC (super fast)
    mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40)

    # Convert to simple vector
    return mfcc.mean(axis=1)   # Output shape = (40,)

print("\n⚡ FAST MFCC EXTRACTION STARTED ⚡\n")

for accent in os.listdir(DATA_DIR):
    accent_path = os.path.join(DATA_DIR, accent)
    if not os.path.isdir(accent_path):
        continue

    print(f"→ Processing {accent}...", end=" ")

    features = []

    for filename in os.listdir(accent_path):
        if not filename.endswith(".wav"):
            continue

        file_path = os.path.join(accent_path, filename)

        try:
            mfcc_vec = extract_mfcc_fast(file_path)
            features.append(mfcc_vec)

        except Exception as e:
            print(f"\n  Skipped bad file: {filename} ({e})")
            continue

    features = np.array(features)

    save_path = os.path.join(OUT_DIR, f"{accent}_mfcc.npy")
    np.save(save_path, features)

    print(f"✔ saved {features.shape}")

print("\n✅ DONE — MFCC features extracted FAST.")
