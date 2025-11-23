import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# True and predicted labels
true_labels = ["andhra_pradesh", "andhra_pradesh"]
predicted_labels = ["jharkhand", "kerala"]

# All class names in your model
classes = ["andhra_pradesh", "gujrat", "jharkhand", "karnataka", "kerala", "tamil"]

# Build confusion matrix
cm = confusion_matrix(true_labels, predicted_labels, labels=classes)

# Plot
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix – Child Speech Generalization Test")
plt.tight_layout()
plt.show()
