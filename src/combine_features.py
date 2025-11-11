import numpy as np
import os

features_dir = r"C:\Users\amish\OneDrive\Desktop\speech-accent-project\data\features"

all_features = []
all_labels = []

for state in os.listdir(features_dir):
    state_path = os.path.join(features_dir, state)
    if not os.path.isdir(state_path):
        continue

    for file_name in os.listdir(state_path):
        if file_name.endswith(".npy"):
            feature_path = os.path.join(state_path, file_name)
            features = np.load(feature_path)

            # --- Fix: take mean over time/frames if 2D or 3D ---
            if features.ndim == 3:
                features = features.mean(axis=0)  # reduce frames
            if features.ndim == 2:
                features = features.mean(axis=0)  # reduce frames

            # now features should be 1D (length = 768)
            features = features.flatten()  # ensure 1D
            all_features.append(features)
            all_labels.append(state)

# Safety check
if len(all_features) == 0:
    raise ValueError("⚠️ No feature files found in features_dir. Check your path!")

# Combine all features
X = np.vstack(all_features)
y = np.array(all_labels)

# Save
np.save(os.path.join(features_dir, "features.npy"), X)
np.save(os.path.join(features_dir, "labels.npy"), y)

print(f"✅ Combined {len(all_labels)} samples into features.npy and labels.npy.")

