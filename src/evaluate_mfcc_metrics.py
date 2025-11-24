import os
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------
# Correct MFCC directory
# -------------------------
MFCC_DIR = "../data/features/mfcc"

# -------------------------
# Model paths
# -------------------------
MODEL_PATH = "./models/accent_classifier.pkl"
ENCODER_PATH = "./models/label_encoder.pkl"

clf = joblib.load(MODEL_PATH)
enc = joblib.load(ENCODER_PATH)

# -------------------------
# Load all MFCC files
# -------------------------
print("📥 Loading MFCC features...\n")

X = []
y = []

for file in os.listdir(MFCC_DIR):
    if not file.endswith("_mfcc.npy"):
        continue

    accent = file.replace("_mfcc.npy", "")
    label_index = enc.transform([accent])[0]

    fpath = os.path.join(MFCC_DIR, file)
    mfcc_features = np.load(fpath)

    for row in mfcc_features:
        X.append(row)
        y.append(label_index)

X = np.array(X)
y = np.array(y)

print("Total MFCC samples loaded:", len(X))

# -------------------------
# Predictions
# -------------------------
y_pred = clf.predict(X)

# Accuracy
acc = accuracy_score(y, y_pred)
print("\n🎯 MFCC Accuracy:", round(acc*100, 2), "%\n")

# Classification report
print("\n📄 Classification Report:\n")
print(classification_report(y, y_pred, target_names=enc.classes_))

# Confusion Matrix
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, cmap="Greens", fmt="d",
            xticklabels=enc.classes_, yticklabels=enc.classes_)
plt.title("MFCC Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
