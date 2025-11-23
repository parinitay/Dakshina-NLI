import os
import soundfile as sf

AUDIO_DIR = r"C:\Users\amish\OneDrive\Desktop\speech-accent-project\data\IndicAccentDB"

bad_files = []
good_files = []

for root, dirs, files in os.walk(AUDIO_DIR):
    for f in files:
        if f.endswith(".wav"):
            path = os.path.join(root, f)
            try:
                data, sr = sf.read(path)

                duration = len(data) / sr if sr > 0 else 0

                if data is None or len(data) < 200:
                    bad_files.append((path, "Too short"))
                elif duration > 30:
                    bad_files.append((path, f"Too long ({duration:.2f}s)"))
                elif len(data.shape) > 1:
                    bad_files.append((path, "Stereo"))
                elif sr not in [16000, 44100, 48000]:
                    bad_files.append((path, f"Odd sample rate ({sr})"))
                else:
                    good_files.append(path)

            except Exception as e:
                bad_files.append((path, f"Corrupted: {str(e)}"))

print("\n=== BAD FILES FOUND ===")
for bf in bad_files:
    print(bf)

print("\nTotal bad files:", len(bad_files))
print("Total good files:", len(good_files))
