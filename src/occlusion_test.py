# occlusion_test.py
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Paths
FEATURE_PATH = r"C:\Users\amish\OneDrive\Desktop\speech-accent-project\data\features\features.npy"
LABEL_PATH   = r"C:\Users\amish\OneDrive\Desktop\speech-accent-project\data\features\labels.npy"
MODEL_PATH   = r"C:\Users\amish\OneDrive\Desktop\speech-accent-project\src\models\accent_classifier.pkl"
ENCODER_PATH = r"C:\Users\amish\OneDrive\Desktop\speech-accent-project\src\models\label_encoder.pkl"

# Load everything
X = np.load(FEATURE_PATH)
y = np.load(LABEL_PATH)
enc = joblib.load(ENCODER_PATH)
clf = joblib.load(MODEL_PATH)

y_enc = enc.transform(y)
_, X_test, _, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)

# Base accuracy
base_acc = accuracy_score(y_test, clf.predict(X_test))
print("Base Accuracy:", base_acc)

# Occlusion test on first 20 dimensions
drops = []

for dim in range(20):
    X_mod = X_test.copy()
    X_mod[:, dim] = 0  # remove 1 dimension
    acc = accuracy_score(y_test, clf.predict(X_mod))
    drop = base_acc - acc
    drops.append((dim, drop))
    print(f"Dim {dim}: Accuracy Drop = {drop:.4f}")

# Print most important dims
print("\nMost important dimensions (bigger drop = more interpretable):")
for dim, drop in sorted(drops, key=lambda x: x[1], reverse=True):
    print(f"Dimension {dim}: Drop = {drop:.4f}")
