🎙️ Dakshina — Native Language Identification of Indian English Speakers
Dakshina is an AI-powered Accent Classification system that identifies regional Indian English accents and recommends traditional cuisines from the detected region.  
It uses HuBERT speech embeddings and a Logistic Regression classifier, wrapped in a modernStreamlit web applicatio
 Features
-  Accent Detection from uploaded or recorded audio  
-  Cuisine Recommendation based on predicted region  
-  Uses HuBERT (facebook/hubert-base-ls960) for speech embeddings  
- Clean evaluation metrics (accuracy, confusion matrix, F1-score)  
-  Fully interactive Streamlit UI    

## 📁 Project Structure
Dakshina-NLI/
│
├── data/
│ ├── IndicAccentDB/ # HuggingFace dataset
│ ├── features/ # Extracted MFCC / HuBERT features
│ └── test_audio/ # Sample audio for testing
│
├── models/
│ ├── accent_model.pkl # Trained logistic regression model
│ ├── label_encoder.pkl # Label encoder for accents
│
├── src/
│ ├── app.py # Streamlit full application
│ ├── visuals.py # MFCC / HuBERT visualizations
│ ├── evaluate.py # Evaluation & metrics
│ ├── train_classifier.py # Training script
│ └── utils/ # Preprocessing utilities
│
├── requirements.txt
└── README.md

📦 Installation & Setup
1️⃣ Clone the Repository
```sh
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

▶️ Running the Application
Start Streamlit App
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
