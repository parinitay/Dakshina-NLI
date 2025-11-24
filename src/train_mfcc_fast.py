import os
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

MFCC_DIR = "../data/features/mfcc"
MODEL_OUT = "./mfcc_models"

os.makedirs(MODEL_OUT, exist_ok=True)

X = []
y = []

print("⚡ Loading MFCC features FAST...")

for file in os.listdir(MFCC_DIR):
    if file.endswith("_mfcc.npy"):
        label = file.replace("_mfcc.npy", "")
        data = np.load(os.path.join(MFCC_DIR, file))

        X.append(data)  # (N,40)
        y += [label] * len(data)

X = np.vstack(X)
y = np.array(y)

print("Total samples:", len(X))

encoder = LabelEncoder()
y_enc = encoder.fit_transform(y)

print("⚡ Training Logistic Regression...")
clf = LogisticRegression(max_iter=1500, n_jobs=-1).fit(X, y_enc)

joblib.dump(clf, f"{MODEL_OUT}/mfcc_classifier.pkl")
joblib.dump(encoder, f"{MODEL_OUT}/mfcc_label_encoder.pkl")

print("\n✔ MFCC Model trained FAST and saved!")
