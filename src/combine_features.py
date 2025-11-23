import numpy as np
import os

# Path where all state folders are stored
features_dir = r"C:\Users\amish\OneDrive\Desktop\speech-accent-project\data\features"

all_features = []
all_labels = []

# Loop through each state folder
for state in os.listdir(features_dir):
    state_path = os.path.join(features_dir, state)

    # Skip files, only process directories
    if not os.path.isdir(state_path):
        continue

    # Loop through .npy files inside each state folder
    for file_name in os.listdir(state_path):
        if file_name.endswith(".npy"):
            feature_path = os.path.join(state_path, file_name)
            features = np.load(feature_path)

            # Reduce shape if 2D or 3D → convert to 1D
            if features.ndim == 3:
                features = features.mean(axis=0)
            if features.ndim == 2:
                features = features.mean(axis=0)

            features = features.flatten()   # ensure 1D
            all_features.append(features)
            all_labels.append(state)

# Safety check
if len(all_features) == 0:
    raise ValueError("⚠️ No feature files found in features_dir. Check your path!")

# Convert to arrays
X = np.vstack(all_features)
y = np.array(all_labels)

# Save combined feature & label files
np.save(os.path.join(features_dir, "features.npy"), X)
np.save(os.path.join(features_dir, "labels.npy"), y)

print(f"✅ Combined {len(all_labels)} samples into features.npy and labels.npy.")
