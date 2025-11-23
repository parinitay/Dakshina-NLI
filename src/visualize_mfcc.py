import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 🎧 Load your audio file
audio_path = "data/test_audio/new_audio.wav"
 # <-- change this to your actual path
y, sr = librosa.load(audio_path, sr=None)

# 🎼 Extract MFCC features
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# 🧠 Print shape info
print("MFCC shape:", mfccs.shape)

# 🔥 Plot MFCC as a heatmap
plt.figure(figsize=(10, 6))
librosa.display.specshow(mfccs, x_axis='time', sr=sr, cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title("MFCC Heatmap")
plt.xlabel("Time")
plt.ylabel("MFCC Coefficients")
plt.tight_layout()
plt.show()
