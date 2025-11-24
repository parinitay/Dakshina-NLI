import os
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

MFCC_DIR = "../data/features/mfcc"
MODEL_DIR = "./mfcc_models"

clf = joblib.load(f"{MODEL_DIR}/mfcc_classifier.pkl")
encoder = joblib.load(f"{MODEL_DIR}/mfcc_label_encoder.pkl")

X = []
y = []

print("⚡ Loading MFCC features...\n")

for file in os.listdir(MFCC_DIR):
    if file.endswith("_mfcc.npy"):
        label = file.replace("_mfcc.npy", "")
        arr = np.load(os.path.join(MFCC_DIR, file))

        X.append(arr)
        y += [label] * len(arr)

X = np.vstack(X)
y = np.array(y)
y_enc = encoder.transform(y)

print("Total MFCC samples:", len(X))

# Predict fast
y_pred = clf.predict(X)

# Metrics
print("\n🎯 Accuracy:", round(accuracy_score(y_enc, y_pred) * 100, 2), "%\n")
print(classification_report(y_enc, y_pred, target_names=encoder.classes_))

# Confusion matrix
cm = confusion_matrix(y_enc, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, cmap="Greens",
            xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title("MFCC Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
