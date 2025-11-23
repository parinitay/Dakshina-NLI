import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

X_words = np.load("../data/word_X.npy")
X_sent = np.load("../data/sent_X.npy")

avg_word_len = X_words.shape[1]
avg_sent_len = X_sent.shape[1]

word_variance = X_words.var()
sent_variance = X_sent.var()

similarity = cosine_similarity(X_words.mean(axis=0).reshape(1, -1),
                               X_sent.mean(axis=0).reshape(1, -1))[0][0]

print("----- FEATURE COMPARISON -----")
print("Word Embedding Size:", avg_word_len)
print("Sentence Embedding Size:", avg_sent_len)
print()
print("Word Feature Variance:", word_variance)
print("Sentence Feature Variance:", sent_variance)
print()
print("Cosine Similarity Words vs Sentences:", similarity)
print("--------------------------------")
