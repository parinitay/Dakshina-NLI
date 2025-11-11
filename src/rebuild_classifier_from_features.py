import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Paths
FEATURES_DIR = r"C:\Users\amish\OneDrive\Desktop\speech-accent-project\data\features"
MODEL_DIR = r"C:\Users\amish\OneDrive\Desktop\speech-accent-project\src\models"

# Load combined features
X_path = os.path.join(FEATURES_DIR, "features.npy")
y_path = os.path.join(FEATURES_DIR, "labels.npy")

if not os.path.exists(X_path) or not os.path.exists(y_path):
    raise FileNotFoundError("⚠️ features.npy or labels.npy not found. Run combine_features.py first!")

X = np.load(X_path)
y = np.load(y_path)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Retrain only classifier (fast)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Rebuilt Classifier Accuracy: {round(acc * 100, 2)}%")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save again
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(clf, os.path.join(MODEL_DIR, "accent_classifier.pkl"))
joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))

print(f"\n🎉 Rebuilt model saved to {MODEL_DIR}")
