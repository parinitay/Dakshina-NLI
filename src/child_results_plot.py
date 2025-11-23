import matplotlib.pyplot as plt

true_accents = ["Andhra Pradesh", "Andhra Pradesh"]
predicted_accents = ["Jharkhand", "Kerala"]
files = ["child1_male", "child2_female"]

plt.figure(figsize=(8, 5))
plt.bar(files, [0, 0], label="True Accent (Andhra Pradesh)")  
plt.bar(files, [1, 1], label="Predicted Accent: Jharkhand / Kerala")

plt.title("Model Performance on Child Speech Samples")
plt.xlabel("Child Audio File")
plt.ylabel("Accent Match (1=Correct, 0=Wrong)")
plt.ylim(0, 1.2)

plt.legend()
plt.show()
