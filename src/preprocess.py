# src/preprocess.py
import os
import pandas as pd

def generate_metadata(data_dir="data/IndicAccentDB", out_csv="data/metadata.csv"):
    rows = []
    print("🎧 Scanning local dataset folders...")

    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".wav") or f.endswith(".mp3"):
                accent = os.path.basename(root)  # folder name = label
                path = os.path.join(root, f)
                rows.append({"path": path, "label": accent})

    if not rows:
        raise ValueError("❌ No audio files found! Check your folder path.")

    df = pd.DataFrame(rows)
    os.makedirs("data", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"✅ Metadata created and saved to {out_csv} ({len(df)} entries).")

if __name__ == "__main__":
    generate_metadata()


