# ---------- train_accent_classifier.py ----------
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Paths
FEATURE_DIR = r"C:\Users\amish\OneDrive\Desktop\speech-accent-project\features"

# Load all state embeddings
X, y = [], []
for fname in os.listdir(FEATURE_DIR):
    if fname.endswith("_features.npy"):
        state = fname.replace("_features.npy", "")
        arr = np.load(os.path.join(FEATURE_DIR, fname))
        X.append(arr)
        y += [state] * arr.shape[0]

X = np.vstack(X)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Train Random Forest
print("🌲 Training Random Forest model...")
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
print("\n🧾 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model + label encoder
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/accent_classifier.pkl")
joblib.dump(le, "models/label_encoder.pkl")
print("\n💾 Model and label encoder saved!")
