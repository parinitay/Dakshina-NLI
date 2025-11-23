import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ============================
# ABSOLUTE PATHS
# ============================
FEATURE_PATH = r"C:\Users\amish\OneDrive\Desktop\speech-accent-project\data\features\features.npy"
LABEL_PATH   = r"C:\Users\amish\OneDrive\Desktop\speech-accent-project\data\features\labels.npy"
MODEL_PATH   = r"C:\Users\amish\OneDrive\Desktop\speech-accent-project\src\models\accent_classifier.pkl"
ENCODER_PATH = r"C:\Users\amish\OneDrive\Desktop\speech-accent-project\src\models\label_encoder.pkl"

# ============================
# LOAD DATA, MODEL, ENCODER
# ============================
print("📥 Loading features, model, and encoder...")

X = np.load(FEATURE_PATH)
y = np.load(LABEL_PATH)
clf = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)  # FIX ADDED

print(f"✔ Features Loaded: {X.shape}")
print(f"✔ Labels Loaded:   {y.shape}")

# ============================
# ENCODE LABELS PROPERLY
# ============================
y_encoded = encoder.transform(y)   # convert strings → numbers

# ============================
# TRAIN/TEST SPLIT
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# ============================
# PREDICT
# ============================
print("\n🎯 Evaluating Model...")
y_pred = clf.predict(X_test)

# ============================
# ACCURACY
# ============================
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc*100:.2f}%\n")

# ============================
# CLASSIFICATION REPORT
# ============================
print("📄 Classification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# ============================
# CONFUSION MATRIX
# ============================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap="Greens", fmt="d",
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_)
plt.title("Accent Classification - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
