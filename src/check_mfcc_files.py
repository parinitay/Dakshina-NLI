import os
import numpy as np

DIR = "../data/features"

print("Checking MFCC feature files...\n")

for f in os.listdir(DIR):
    if f.endswith("_features.npy"):
        path = os.path.join(DIR, f)
        arr = np.load(path)
        print(f"{f} → shape {arr.shape}")
