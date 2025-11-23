import matplotlib.pyplot as plt
import numpy as np

# ==== ENTER YOUR VALUES HERE ====
accuracy = 0.9959
robustness_word = 0.01348946
robustness_sentence = 0.008917565
interpretability = 0.9048491

# ========== Create Matrix ==========
data = [
    ["Metric", "Score"],
    ["Accuracy", f"{accuracy:.4f}"],
    ["Robustness (Word)", f"{robustness_word:.6f}"],
    ["Robustness (Sentence)", f"{robustness_sentence:.6f}"],
    ["Interpretability", f"{interpretability:.4f}"]
]

fig, ax = plt.subplots(figsize=(6, 4))
ax.axis("off")

table = ax.table(
    cellText=data,
    loc="center",
    cellLoc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)

# Green outline
for key, cell in table.get_celld().items():
    cell.set_edgecolor("green")
    cell.set_linewidth(1.5)

plt.title("Accent Model Metrics (Confusion-Matrix Style)", fontsize=15)
plt.show()
