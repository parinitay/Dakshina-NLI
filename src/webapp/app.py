import streamlit as st
import torch
import torchaudio
import numpy as np
import joblib
from transformers import Wav2Vec2FeatureExtractor, HubertModel
import tempfile
import os

torchaudio.set_audio_backend("soundfile")

# ================== STREAMLIT PAGE CONFIG ==================
st.set_page_config(
    page_title="Dakshina ",
    page_icon="🎙️",
    layout="centered",
)

# ================== CUSTOM STYLING ==================
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Texturina:wght@400;600;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

        [data-testid="stAppViewContainer"] {
            background-color: #FEFAE0 !important;
            color: #5F6F52;
            font-family: 'Texturina', serif;
        }

        [data-testid="stHeader"] {
            background: none !important;
        }

        [data-testid="stSidebar"] {
            background-color: #5F6F52 !important;
            backdrop-filter: blur(12px);
            border-right: 2px solid rgba(87, 73, 100, 0.2);
        }

        .title {
            font-family: 'Texturina', serif;
            text-align: center;
            font-size: 52px;
            font-weight: 700;
            color: #5F6F52;
            margin-top: 15px;
        }

        .subtitle {
            text-align: center;
            color: #A9B388;
            font-family: 'Poppins', sans-serif;
            margin-bottom: 40px;
            font-size: 18px;
        }

        /* Box Styles */
        .input-box, .result-box {
            background-color: #F5E6C8;
            color: #5F6F52;
            border-radius: 25px;
            padding: 30px;
            width: 100%;
            max-width: 800px;
            margin: 0 auto 40px auto;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.1);
            border: 2px solid #E6CBA8;
        }

        /* Buttons */
        .stButton>button {
            background-color: transparent;
            border: 2px solid #5F6F52;
            color: #5F6F52;
            font-weight: bold;
            border-radius: 25px;
            padding: 10px 25px;
            transition: 0.3s ease;
        }

        .stButton>button:hover {
            background-color: #5F6F52;
            color: #FEFAE0;
        }

        /* 🎯 Fix radio button text visibility */
        [data-testid="stRadio"] label p {
            color: #000000 !important;
            font-weight: 600 !important;
        }

        [data-testid="stRadio"] label span {
            color: #000000 !important;
            font-weight: 600 !important;
        }

        /* Hover fix */
        [data-testid="stRadio"] label:hover p {
            color: #000000 !important;
        }

    </style>
""", unsafe_allow_html=True)

# ================== MODEL PATHS ==================
MODEL_PATH = "src/models/accent_classifier.pkl"
ENCODER_PATH = "src/models/label_encoder.pkl"

# ================== LOAD MODELS ==================
@st.cache_resource
def load_models():
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960", torch_dtype=torch.float32)
    hubert.eval()
    clf = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    return feature_extractor, hubert, clf, le

try:
    feature_extractor, hubert, clf, le = load_models()
    model_loaded = True
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    model_loaded = False

device = "cuda" if torch.cuda.is_available() else "cpu"

# ================== CUISINE SUGGESTIONS ==================
cuisine_suggestions = {
    "andhra_pradesh": ["Pesarattu 🥬", "Gongura Pachadi 🍀", "Pulihora 🍚", "Kodi Kura 🍗"],
    "tamil": ["Dosa 🫓", "Idli ⚪", "Chettinad Chicken 🍲 ", "Pongal 🧉"],
    "kerala": ["Appam with Stew 🍲 ", "Puttu and Kadala Curry 🍲", "Fish Moilee 🐟", "Parotta with Beef Fry 🫓"],
    "karnataka": ["Bisi Bele Bath 🍛", "Ragi Mudde 𓎩 ", "Mysore Pak 🧈", "Neer Dosa 🫓"],
    "gujrat": ["Dhokla 🧽", "Undhiyu 🍲", "Thepla 🫓   ", "Khandvi 🥖"],
    "jharkhand": ["Thekua 🍪", "Litti Chokha 🫓", "Dhuska 🫓", "Handia 🧉"]
}

# ================== PREDICTION FUNCTION ==================
def predict_accent(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    waveform = waveform.mean(dim=0, keepdim=True)
    waveform = torchaudio.functional.resample(waveform, sr, 16000)

    inputs = feature_extractor(
        waveform.squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = hubert(inputs.input_values.to(device))
        hidden = outputs.last_hidden_state.mean(dim=1).cpu().numpy().astype(np.float16)

    pred = clf.predict(hidden)
    label = le.inverse_transform(pred)[0]

    key = label.lower().replace(" ", "_")
    cuisines = cuisine_suggestions.get(key, ["No cuisines available"])
    cuisine_html = "<br>".join([f"🍴 {c}" for c in cuisines])
    return label.capitalize(), cuisine_html

# ================== MAIN APP ==================
st.sidebar.title("🌸 Dakshina")
page = st.sidebar.radio("Navigate", ["🏠 Home", "📜 About Us"])

# ================== HOME PAGE ==================
if page == "🏠 Home":
    st.markdown("<div class='title'>🌿 Dakshina</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Every hello has a hometown.</div>", unsafe_allow_html=True)

    choice = st.radio("🎤 Choose input method:", ["📁 Upload Audio File", "🎙️ Record Audio"], horizontal=True)

    audio_path = None
    if choice == "📁 Upload Audio File":
        uploaded = st.file_uploader("Upload an audio file (.wav)", type=["wav"])
        if uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded.read())
                audio_path = tmp.name
    elif choice == "🎙️ Record Audio":
        audio_data = st.audio_input("Record your voice:")
        if audio_data:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_data.getbuffer())
                audio_path = tmp.name

    if st.button("🔍 Predict Accent"):
        if audio_path:
            with st.spinner("Analyzing your voice..."):
                label, cuisine_html = predict_accent(audio_path)
            st.markdown(
                f"""
                <div class='result-box'>
                    <h3>Predicted Accent: <b>{label}</b></h3>
                    <h4>Suggested Cuisine:</h4>
                    <p>{cuisine_html}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            os.remove(audio_path)
        else:
            st.warning("⚠️ Please upload or record an audio file first!")

# ================== ABOUT US PAGE ==================
elif page == "📜 About Us":
    st.markdown("<div class='title'> About Dakshina</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='input-box'>
    <p><b>Dakshina</b> was created out of a simple curiosity — 
        how beautifully the sound of language changes across regions, even within the same country.  
        In India, accents are not just ways of speaking; they’re echoes of culture, community, and emotion.</p>

    <p>This project explores that connection between <b>voice and identity</b> — 
        using the power of machine learning to recognize regional Indian English accents.  
        Once identified, Dakshina celebrates the culture by suggesting traditional cuisines from that region 🍛</p>

    <p>Built using <b>HuBERT</b> (Hidden-Unit BERT) for extracting rich speech features, 
    combined with a <b>Logistic Regression</b> classifier, Dakshina brings together technology, language, and culture — 
    wrapped in a calm, minimal design.</p>
                
    <hr style="border:1px solid #E6CBA8; margin:30px 0;">

    <p style="text-align:center;"> 
    <br><br><b>— Team Dakshina, 2025 💫</b>
    </p>

    <p><b> By Umasri , Parinita , Srishma </b></p>
    </div>
    """, unsafe_allow_html=True)
