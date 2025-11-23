# Dakshina-NLI
Dakshina is an AI-powered app that identifies Indian English accents using HuBERT and Logistic Regression, trained on the IndicAccentDB dataset from Hugging Face. Built with Python, PyTorch, and Streamlit, it predicts accents from six Indian states and suggests their regional cuisines.

Features
✔ Accent Classification (6 Indian Accents)

Andhra Pradesh

Gujarat

Jharkhand

Karnataka

Kerala

Tamil Nadu

✔ Complete Feature Extraction

HuBERT embedding extraction

MFCC extraction

Layer-wise HuBERT feature analysis

✔ Evaluation + Analysis

Accuracy / F1-score

Confusion Matrices

Word-vs-Sentence Accent Comparison

Child Generalization Test

Robustness scoring

Interpretability scoring

✔ Visualizations

Layer-wise accuracy plot

Word vs sentence barplot

Confusion matrix plots

Robustness & interpretability graphs



📁 Project Structure

<img width="1024" height="1024" alt="ChatGPT Image Nov 24, 2025, 12_15_15 AM" src="https://github.com/user-attachments/assets/812c1b8a-b9ee-4052-8050-bdf9eeaf96c1" />


🔧 Installation
1. Clone the repository
git clone https://github.com/parinitay/Dakshina-NLI.git
cd Dakshina-NLI

2. Create & activate virtual environment
python -m venv venv
venv\Scripts\activate      # Windows

3. Install dependencies
pip install -r requirements.txt


(Dependencies include: PyTorch, HuggingFace Transformers, librosa, sklearn, matplotlib, seaborn, pandas, soundfile.)

🎤 Dataset Setup

Place your dataset inside:

data/IndicAccentDB/
    ├── andhra_pradesh/
    ├── gujrat/
    ├── jharkhand/
    ├── karnataka/
    ├── kerala/
    └── tamil/


Each folder must contain .wav files.

Audio recommended: 16 kHz, Mono

🧠 Feature Extraction
Extract HuBERT Features
python src/hubert_feature_extraction.py

Extract MFCC Features
python src/features_mfcc.py


Outputs go to:

data/features/

🎯 Train Model
python src/train_classifier_from_features.py


Model saved in:

models/accent_classifier.pkl
models/label_encoder.pkl

🧪 Evaluate Model
python src/evaluate_model.py


Generates:

Accuracy

Precision / Recall / F1

Confusion Matrix

Bar Plots

🔍 HuBERT Layer-wise Analysis

To find which layer captures accent information best:

python src/layer_analysis_clean.py


Generates:

Layer-wise accuracy list

Best performing layer

Layer accuracy plot (via plot_layer_accuracy.py)

🗣 Word-Level vs Sentence-Level Experiments
Setup folders:
data/word_samples/andhra_pradesh/
data/sentence_samples/andhra_pradesh/

Extract features:
python src/extract_words_sentences.py

Evaluate:
python src/evaluate_words_vs_sentences.py

Confusion matrix:
python src/words_sentences_confusion.py

Visualization:
python src/words_sentences_barplot.py

👶 Child Generalization Test

Place child audio files in:

data/child_test/
    child1_male.wav
    child2_female.wav


Run:

python src/test_child_generalization.py


Confusion matrix:

python src/child_confusion_matrix.py


Plot:

python src/child_results_plot.py

🛡 Robustness & Interpretability
Robustness visualization
python src/robustness_visualization.py

Interpretability visualization
python src/interpretability_plot.py


Produces:

Feature stability plots

Cosine similarity graphs

Layer-wise interpretability visualization

📦 Requirements

To regenerate requirements:

pip freeze > requirements.txt


Your environment includes:

torch

torchaudio

transformers

librosa

scikit-learn

soundfile

pandas

seaborn

matplotlib

streamlit

outputs
<img width="1918" height="917" alt="image" src="https://github.com/user-attachments/assets/4680e79b-c5d9-4ffc-ab04-ae1c5330c617" />

<img width="1913" height="925" alt="image" src="https://github.com/user-attachments/assets/cf0a8b0b-a724-494c-bfd6-5c0790d57717" />

<img width="1912" height="927" alt="image" src="https://github.com/user-attachments/assets/c9d403e3-b711-4cff-9c70-abe99825b8c1" />

