import matplotlib.pyplot as plt

# Your interpretability score
cosine_similarity = 0.9048491   # replace with your final value

plt.figure(figsize=(7, 6))

bars = plt.bar(["Interpretability Score"], 
                [cosine_similarity], 
                color="green",
                alpha=0.85)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, 
             height + 0.02, 
             f"{height:.4f}",
             ha='center', va='bottom', fontsize=12)

plt.title("Interpretability Visualization (Cosine Similarity)", fontsize=16)
plt.ylabel("Similarity (Higher = Better)", fontsize=13)
plt.ylim(0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()
