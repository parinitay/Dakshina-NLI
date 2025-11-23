import matplotlib.pyplot as plt

# Replace with your final values
layers = list(range(13))
accuracies = [
    0.9979,
    0.9979,
    1.0000,
    0.9959,
    0.9959,
    0.9979,
    0.9959,
    0.9959,
    0.9959,
    0.9938,
    0.9959,
    0.9979,
    0.9990  # Replace with actual Layer 12 when done
]

plt.figure(figsize=(10, 5))
plt.plot(layers, accuracies, marker='o')
plt.xticks(layers)
plt.xlabel("HuBERT Layer")
plt.ylabel("Accuracy")
plt.title("Layer-wise Accent Classification Accuracy")
plt.grid(True)
plt.tight_layout()
plt.show()
