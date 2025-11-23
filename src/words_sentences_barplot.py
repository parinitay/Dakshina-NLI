import matplotlib.pyplot as plt

word_acc = 1.0   # correct because all your word samples predicted Andhra
sent_acc = 1.0   # if both sentences predicted Andhra (change if needed)

labels = ["Word-Level", "Sentence-Level"]
values = [word_acc*100, sent_acc*100]

plt.figure(figsize=(6,5))
plt.bar(labels, values)
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Comparison: Word vs Sentence Level")
plt.ylim(0, 100)
plt.show()
