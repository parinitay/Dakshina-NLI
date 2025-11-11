import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load features and labels
features = np.load("../data/features/features.npy")
labels = np.load("../data/features/labels.npy")

print(f"✅ Loaded features: {features.shape}, labels: {labels.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train classifier
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("✅ Evaluation Results:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model and scaler
import joblib
joblib.dump(model, "../models/accent_model.pkl")
joblib.dump(scaler, "../models/scaler.pkl")

print("✅ Model and scaler saved to ../models/")
