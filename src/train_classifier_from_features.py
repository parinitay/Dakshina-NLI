# src/train_classifier_from_features.py
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
import joblib

# -------------------------------------------------------
# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = os.path.join(BASE_DIR, "../data/features")
MODEL_PATH = os.path.join(BASE_DIR, "models/accent_classifier.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models/label_encoder.pkl")

# -------------------------------------------------------
# Load saved embeddings
X = np.load(os.path.join(FEATURES_DIR, "features.npy"))
y = np.load(os.path.join(FEATURES_DIR, "labels.npy"))

print(f"✅ Loaded features: {X.shape}, labels: {y.shape}")

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
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(clf, MODEL_PATH)
joblib.dump(label_encoder, ENCODER_PATH)

print(f"\n✅ Model saved at: {MODEL_PATH}")
print(f"✅ Label encoder saved at: {ENCODER_PATH}")
print("\n🎉 Training complete!")
