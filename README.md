🎙️ Dakshina — Native Language Identification of Indian English Speakers
Dakshina is an AI-powered Accent Classification system that identifies regional Indian English accents and recommends traditional cuisines from the detected region.  
It uses HuBERT speech embeddings and a Logistic Regression classifier, wrapped in a modernStreamlit web application
 Features
  Accent Detection from uploaded or recorded audio  
  Cuisine Recommendation based on predicted region  
  Uses HuBERT (facebook/hubert-base-ls960) for speech embeddings  
  Clean evaluation metrics (accuracy, confusion matrix, F1-score)  
  Fully interactive Streamlit UI    

 🗂️ Project Structure
 
<img width="1000" height="4000" alt="output-onlinetools (2)" src="https://github.com/user-attachments/assets/216e5cbc-637a-4dbf-b00d-b02ca8e56d65" />




📦 Installation & Setup
1️⃣ Clone the Repository

git clone https://github.com/parinitay/Dakshina-NLI.git
cd Dakshina-NL

2️⃣ Create a Virtual Environment
python -m venv venv

3️⃣ Activate the Environment
Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

4️⃣ Install Dependencies
pip install -r requirements.txt

> IMPORT DATASET

Download IndicAccentDB dataset from [HuggingFace.](https://huggingface.co/datasets/DarshanaS/IndicAccentDb)

Place all audio files in the data/ folder according to your structure.

Ensure labels correspond to:
andhra, kerala, karnataka, tamil, gujarat, jharkhand


> PROJECT EXECUTION PIPELINE

python preprocess.py	              Cleans dataset, filters corrupted audio, normalizes sampling rate

python check_audio_quality.py    	 Detects corrupted or unreadable audio

python make_clean_list.py         	Creates clean audio list used in all further processing

> mfcc Feature Extraction

python src/extract_mfcc_features.py



> HuBERT Feature Extraction

 python src/hubert_feature_extraction.py

> EVALUTION

  python src/evaluate_model.py




> MODEL TRAINING

Training directly from audio

python src/train_classifier.py             	



> Training from saved MFCC features

python src/train_classifier_from_features.py


> ACCURACY, METRICS, CONFUSION MATRIX

python evaluate.py

python src/confusion_style_metrics.py


> WORD vs SENTENCE ANALYSIS

python src/evaluate_words_vs_sentences.py


> GENERALIZATION TESTS (CHILDREN AUDIO)

python src/test_child_generalization.py



> ROBUSTNESS & INTERPRETABILITY

python src/layer_analysis_clean.py

python src/interpretability_plot.py

python src/robustness_visualization.py

> VISUALIZATIONS

python src/visualize_mfcc.py
python src/visualize_hubert.py
python src/visuals.py


> FINAL ACCENT PREDICTION

python src/predict_accent.py


▶️ Running the Application
> Start Streamlit App

streamlit run src/app.py
The web interface will open automatically on:
http://localhost:8501

📊 Visualizations (included in visuals.py)
MFCC Heatmap
Waveform Plot
Spectrogram
HuBERT Embedding Heatmap
Run:
python src/visuals.py


OUTPUT 

<img width="1919" height="917" alt="Screenshot 2025-11-18 185606" src="https://github.com/user-attachments/assets/c02ccfae-5677-42a5-8e26-958ff6135685" />

<img width="1917" height="929" alt="Screenshot 2025-11-18 185657" src="https://github.com/user-attachments/assets/6c7dffca-8d16-4d7e-bee5-97c3e5d34fb2" />

<img width="1914" height="925" alt="Screenshot 2025-11-18 185721" src="https://github.com/user-attachments/assets/6de98fe8-5d3c-4cf4-9ab4-241ac5154cd2" />

<img width="1912" height="927" alt="Screenshot 2025-11-18 185759" src="https://github.com/user-attachments/assets/48a15ca9-6d41-4297-b678-6674dc401f78" />




