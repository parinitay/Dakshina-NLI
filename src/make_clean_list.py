import os
import soundfile as sf

AUDIO_DIR = r"C:\Users\amish\OneDrive\Desktop\speech-accent-project\data\IndicAccentDB"
LABELS = ["andhra_pradesh", "gujrat", "jharkhand", "karnataka", "kerala", "tamil"]

clean_paths = []

for label_idx, label in enumerate(LABELS):
    folder = os.path.join(AUDIO_DIR, label)
    if not os.path.isdir(folder):
        continue

    for f in os.listdir(folder):
        if f.endswith(".wav"):
            path = os.path.join(folder, f)
            try:
                data, sr = sf.read(path)

                # SKIP BAD
                if data is None: 
                    continue
                if len(data) < 200: 
                    continue
                if len(data.shape) > 1:  # stereo
                    continue
                if sr < 8000 or sr > 48000: 
                    continue

                clean_paths.append((path, label_idx))

            except:
                continue

print("GOOD FILES:", len(clean_paths))

with open("clean_audio_list.txt", "w") as f:
    for path, label_idx in clean_paths:
        f.write(path + "|" + str(label_idx) + "\n")

print("Saved clean_audio_list.txt")
