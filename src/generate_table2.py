import pandas as pd

# ----------------------------
# ENTER YOUR ACCURACIES HERE
# ----------------------------
hubert_accuracy = 0.98   # <---- replace with yours
mfcc_accuracy = 0.72     # <---- replace with yours

# ----------------------------
# CREATE TABLE
# ----------------------------
data = {
    "Method": ["HuBERT (768-dim)", "MFCC (40-dim)"],
    "Accuracy": [hubert_accuracy, mfcc_accuracy],
    "Robustness": ["High", "Medium"],
    "Interpretability": ["Medium", "High"],
}

df = pd.DataFrame(data)

# ----------------------------
# SAVE TABLE AS CSV & MD
# ----------------------------
df.to_csv("table2_hubert_vs_mfcc.csv", index=False)

# Save Markdown table
with open("table2_hubert_vs_mfcc.md", "w") as f:
    f.write(df.to_markdown(index=False))

print("✅ Table generated!")
print(df)
