import matplotlib.pyplot as plt
import numpy as np

# --------------------------
# ENTER YOUR ROBUSTNESS SCORES HERE
# --------------------------
word_robustness = 0.01348946     # example values
sentence_robustness = 0.008917565

# --------------------------
# DATA FOR PLOT
# --------------------------
labels = ['Word-Level', 'Sentence-Level']
scores = [word_robustness, sentence_robustness]

# --------------------------
# BAR CHART
# --------------------------
plt.figure(figsize=(8, 6))
bars = plt.bar(labels, scores)

# Add values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height,
             f"{height:.4f}",
             ha='center', va='bottom', fontsize=12)

plt.title("Robustness Comparison: Word vs Sentence", fontsize=16)
plt.ylabel("Variance (Lower = More Robust)", fontsize=13)
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
