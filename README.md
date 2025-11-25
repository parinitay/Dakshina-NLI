🎙️ Dakshina — Native Language Identification of Indian English Speakers
Dakshina is an AI-powered Accent Classification system that identifies regional Indian English accents and recommends traditional cuisines from the detected region.  
It uses HuBERT speech embeddings and a Logistic Regression classifier, wrapped in a modernStreamlit web application
  

 🗂️ Project Structure
 

```bash
📦 speech-accent-project/
│
├── 📂 data/
│   ├── 📂 IndicAccentDB/
│   ├── 📂 features/
│
├── 📂 src/
│   ├── 📄 hubert_feature_extraction.py
│   ├── 📄 extract_mfcc_features.py
│   ├── 📄 combine_features.py
│   ├── 📄 train_classifier_from_features.py
│   ├── 📄 rebuild_classifier_from_features.py
│   ├── 📄 train_mfcc_fast.py
│   ├── 📄 evaluate_model.py
│   ├── 📄 evaluate_mfcc.py
│   ├── 📄 predict_accent.py
│   ├── 📂 webapp/
│   │   └── 📄 app.py
│   ├── 📄 extract_child_features.py
│   ├── 📄 test_child_generalization.py
│   └── 📄 child_generalization_metrics.py
│   ├── 📄 visuals.py
│   ├── 📄 visualize_mfcc.py
│   └── 📄 visualize_hubert.py
│
├── 📂 src/mfcc_models/
│   ├── 📄 mfcc_classifier.pkl
│   └── 📄 mfcc_label_encoder.pkl
│
├── 📂 src/models/
│   ├── 📄 accent_classifier.pkl
│   └── 📄 label_encoder.pkl
│
├── 📄 requirements.txt
└── 📄 README.md

```

📦 Installation & Setup
1️⃣ Clone the Repository

git clone https://github.com/parinitay/Dakshina-NLI.git

```bash
cd Dakshina-NL
```

2️⃣ Create a Virtual Environment


```bash
python -m venv venv
```

3️⃣ Activate the Environment
Windows:

```bash
venv\Scripts\activate
```
Mac/Linux:
source venv/bin/activate

4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```
> IMPORT DATASET

Download IndicAccentDB dataset from [HuggingFace.](https://huggingface.co/datasets/DarshanaS/IndicAccentDb)

Place all audio files in the data/ folder according to your structure.

Ensure labels correspond to:
andhra, kerala, karnataka, tamil, gujarat, jharkhand




> mfcc Feature Extraction

```bash
python src/extract_mfcc_features.py
```



> HuBERT Feature Extraction

```bash
python src/hubert_feature_extraction.py
```


> COMBINE HuBERT FEATURES

```bash
python src/combine_features.py
```

> MODEL TRAINING

Training directly from audio

          	
```bash
python src/train_classifier.py   
```


> Training HuBert Model 

```bash
python src/train_classifier_from_features.py   
```


>Train MFCC Model

```bash
python src/train_mfcc_fast.py
```

🔄 FAST MODEL REBUILD (OPTIONAL)


Sometimes you may want to retrain the Logistic Regression classifier quickly without rerunning HuBERT extraction.
For this purpose, we include:

rebuild_classifier_from_features.py

This script:

Loads the already prepared

data/features/features.npy  
data/features/labels.npy  


Retrains the same Logistic Regression classifier

Saves fresh versions of:

src/models/accent_classifier.pkl
src/models/label_encoder.pkl


Runs in a few seconds (much faster than full extraction)

Run:

```bash
python src/rebuild_classifier_from_features.py
```


Use this script when:

You want to rebuild the model after deleting .pkl files

You made changes to the classifier settings

You are debugging the Streamlit app

You want a fresh model without reprocessing HuBERT embeddings


> EVALUTION FOR HuBERT FEATURES 

  ```bash
 python src/evaluate_model.py
```
  
> EVALUTION FOR mfcc FEATURES

  ```bash
 python evaluate_mfcc_fast.py
```
 

 > FINAL ACCENT PREDICTION

 ```bash
 python src/predict_accent.py
```

 ▶️ Running the Application

   Start the Streamlit app:

  ```bash
streamlit run src/webapp/app.py
```
   

The web interface will open automatically at:

     http://localhost:8501


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🔬 OPTIONAL: DATA CLEANING / EXPERIMENTAL PREPROCESSING


These scripts were used only for experimentation and are not required for the main pipeline.

python preprocess.py
python check_audio_quality.py
python make_clean_list.py


🔬 OPTIONAL: HuBERT LAYER-WISE ANALYSIS (Experimental)

1️⃣ layer_analysis.py

Runs full layer-wise evaluation on the entire dataset.

  ```bash
python src/layer_analysis.py
```

2️⃣ layer_analysis_fast.py

Fast version — evaluates each HuBERT layer using a small random subset (≈80 files/class) and batch processing.

  ```bash
python src/layer_analysis_fast.py
```


3️⃣ layer_analysis_clean.py

Uses a manually cleaned audio list to reduce noise and test layer stability.

  ```bash
python src/layer_analysis_clean.py
```

> EXPERIMENTAL VISUALIZATION SCRIPT

  ```bash
python src/layer_analysis_plot.py
```


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



🎨 VISUALIZATION TOOLS (Optional)

1️⃣ visuals.py (Recommended)

Combined visualization tool that shows:

Waveform

MFCC Heatmap

HuBERT Embedding Heatmap



  ```bash
python src/visuals.py
```


2️⃣ visualize_mfcc.py (Optional)

  ```bash
python src/visualize_mfcc.py
```

3️⃣ visualize_hubert.py (Optional)

  ```bash
python src/visualize_hubert.py
```

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🔬 CHILD SPEECH GENERALIZATION (Optional)

These scripts test how well the trained HuBERT model generalizes to unseen child speech.

1️⃣ extract_child_features.py

Run:

  ```bash
python src/extract_child_features.py
```


2️⃣ test_child_generalization.py

Run:
  ```bash
python src/test_child_generalization.py
```

3️⃣ child_generalization_metrics.py

Run:
  ```bash
python src/child_generalization_metrics.py
```


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🔬 WORD VS SENTENCE LEVEL (Optional)


1️⃣extract_words_sentences.py

  ```bash
python src/extract_words_sentences.py
```

2️⃣evaluate_words_vs_sentences.py

 ```bash
python src/evaluate_words_vs_sentences.py
```


> VISUALS

1️⃣words_sentences_confusion.py

 ```bash
python src/words_sentences_confusion.py
```

2️⃣words_sentences_barplot.py 


 ```bash
python src/words_sentences_barplot.py 
```

DAKSHINA WEBSITE OUTPUT 

<img width="1919" height="917" alt="Screenshot 2025-11-18 185606" src="https://github.com/user-attachments/assets/c02ccfae-5677-42a5-8e26-958ff6135685" />



<img width="1917" height="929" alt="Screenshot 2025-11-18 185657" src="https://github.com/user-attachments/assets/6c7dffca-8d16-4d7e-bee5-97c3e5d34fb2" />




<img width="1914" height="925" alt="Screenshot 2025-11-18 185721" src="https://github.com/user-attachments/assets/6de98fe8-5d3c-4cf4-9ab4-241ac5154cd2" />




<img width="1912" height="927" alt="Screenshot 2025-11-18 185759" src="https://github.com/user-attachments/assets/48a15ca9-6d41-4297-b678-6674dc401f78" />


-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


GitHub Repository link: 
https://github.com/parinitay/Dakshina-NLI 
	Some large files (dataset, checkpoints) could not be pushed due to size limits.


Google Drive link:
https://drive.google.com/file/d/1ImeuadaBP-JGf05-3HIc2GL1ecFdxh_-/view?usp=sharing


