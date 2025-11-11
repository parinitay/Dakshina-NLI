# src/train_classifier.py

import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
import joblib

# ✅ Import the correct function from model_hubert
from model_hubert import get_hubert_embeddings_from_file

# -------------------------------------------------------
# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data/IndicAccentDB")
MODEL_PATH = os.path.join(BASE_DIR, "accent_classifier.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

# -------------------------------------------------------
X = []
y = []

print(f"🎧 Using data folder: {os.path.abspath(DATA_DIR)}")

# Walk through the dataset folders
for root, dirs, files in os.walk(DATA_DIR):
    for file in files:
        if file.lower().endswith(".wav"):
            file_path = os.path.join(root, file)
            language = os.path.basename(os.path.dirname(file_path))
            print(f"Processing: {file_path} | Label: {language}")

            try:
                # ✅ Get HuBERT embeddings directly from file
                emb = get_hubert_embeddings_from_file(file_path)

                if emb is None:
                    print(f"❌ Skipping {file_path} (embedding not generated)")
                    continue

                # Flatten or mean-pool to 1D if needed
                if emb.ndim > 1:
                    emb = emb.mean(axis=0)

                X.append(emb)
                y.append(language)

            except Exception as e:
                print(f"⚠️ Error processing {file_path}: {e}")
                continue

# -------------------------------------------------------
# Validation
if len(X) == 0:
    raise ValueError("❌ No .wav files found or embeddings generated. Check your folder structure and audio files.")

X = np.array(X)
y = np.array(y)

# -------------------------------------------------------
# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# -------------------------------------------------------
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# -------------------------------------------------------
# Train model
print("\n🚀 Training Logistic Regression classifier...")
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# -------------------------------------------------------
# Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n📊 Accuracy:", round(acc * 100, 2), "%")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# -------------------------------------------------------
# Save model & encoder
joblib.dump(clf, MODEL_PATH)
joblib.dump(label_encoder, ENCODER_PATH)

print(f"\n✅ Model saved at: {MODEL_PATH}")
print(f"✅ Label encoder saved at: {ENCODER_PATH}")
print("\n🎉 Training complete!")
