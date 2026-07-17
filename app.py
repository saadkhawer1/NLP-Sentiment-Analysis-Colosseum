import streamlit as st
import joblib
import os
import re
import pandas as pd
from io import BytesIO
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

# 1. Preprocessing logic
NEGATION_WORDS = {'not', 'no', 'never', 'don\'t', "isn't", "aren't", "wasn't", "weren't",
                  "hasn't", "haven't", "hadn't", "doesn't", "didn't", "can't", "couldn't",
                  "shouldn't", "won't", "wouldn't", "mightn't", "mustn't", 'without', 'cannot'}

stop_words = set(stopwords.words('english')) - NEGATION_WORDS
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

# 2. Setup Streamlit UI & Premium Theme
st.set_page_config(page_title="Colosseum Sentiment AI", page_icon="🏛️", layout="centered")

# Custom CSS for Premium Light Theme
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #f8fafc;
        color: #1e293b;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #0f172a !important;
        font-weight: 800 !important;
    }
    
    /* Input Text Area */
    .stTextArea textarea {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05) !important;
        padding: 15px !important;
        font-size: 16px !important;
        color: #334155 !important;
    }
    .stTextArea textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.3) !important;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 15px rgba(37, 99, 235, 0.4) !important;
    }

    /* Result Cards */
    .result-card-pos {
        background-color: #dcfce7;
        border-left: 5px solid #22c55e;
        padding: 20px;
        border-radius: 8px;
        color: #166534;
        font-weight: bold;
        font-size: 18px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05);
        margin-bottom: 20px;
    }
    .result-card-neg {
        background-color: #fee2e2;
        border-left: 5px solid #ef4444;
        padding: 20px;
        border-radius: 8px;
        color: #991b1b;
        font-weight: bold;
        font-size: 18px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05);
        margin-bottom: 20px;
    }
    .result-card-neu {
        background-color: #fef9c3;
        border-left: 5px solid #eab308;
        padding: 20px;
        border-radius: 8px;
        color: #854d0e;
        font-weight: bold;
        font-size: 18px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05);
        margin-bottom: 20px;
    }

    /* History Box */
    .history-box {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgb(0 0 0 / 0.02);
    }
    .history-text { font-style: italic; color: #475569; }
    .history-label { font-weight: 700; margin-top: 5px; }

</style>
""", unsafe_allow_html=True)

# 3. Load Model and Vectorizer
@st.cache_resource
def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(base_dir, "sentiment_model.pkl"))
    vectorizer = joblib.load(os.path.join(base_dir, "tfidf_vectorizer.pkl"))
    return model, vectorizer

try:
    model, vectorizer = load_models()
    is_ready = True
except FileNotFoundError:
    st.error("Model files not found! Please run `python run_pipeline.py` first.")
    is_ready = False

# Session State for History
if "history" not in st.session_state:
    st.session_state.history = []

# Header
st.title("🏛️ Colosseum AI Insight")
st.markdown("<p style='color: #64748b; font-size: 16px;'>Advanced sentiment analysis powered by Machine Learning. Detects Positive, Negative, and Neutral visitor feedback with high precision.</p>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction (CSV)", "History"])

# ================= TAB 1: SINGLE PREDICTION =================
with tab1:
    user_input = st.text_area("Review Input", height=150, placeholder="Example: The Colosseum was breathtaking, but the lines were terribly long and we had to wait for hours.", label_visibility="collapsed")

    if st.button("Analyze Sentiment 🔍") and is_ready:
        if user_input.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing text..."):
                cleaned_input = clean_text(user_input)
                input_vector = vectorizer.transform([cleaned_input])
                prediction = model.predict(input_vector)[0]
                proba = model.predict_proba(input_vector)[0]
                classes = model.classes_
                
                # Save to history
                st.session_state.history.insert(0, {"text": user_input, "pred": prediction})
                if len(st.session_state.history) > 10:
                    st.session_state.history.pop()
                
                st.markdown("### Result")
                if prediction == 'Positive':
                    st.markdown(f"<div class='result-card-pos'>🟢 Positive Sentiment Detected</div>", unsafe_allow_html=True)
                elif prediction == 'Negative':
                    st.markdown(f"<div class='result-card-neg'>🔴 Negative Sentiment Detected</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='result-card-neu'>🟡 Neutral Sentiment Detected</div>", unsafe_allow_html=True)
                    
                st.markdown("**Confidence Score:**")
                cols = st.columns(3)
                for col, cls, prob in zip(cols, classes, proba):
                    col.metric(label=cls, value=f"{prob*100:.1f}%")

# ================= TAB 2: BATCH PREDICTION =================
with tab2:
    st.markdown("### Upload Reviews Dataset")
    st.markdown("Upload a CSV file containing a column named **`text`** (or `review`). The AI will process thousands of reviews at once and let you download the results.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None and is_ready:
        df = pd.read_csv(uploaded_file)
        
        # Find the text column
        text_col = None
        for col in df.columns:
            if 'text' in col.lower() or 'review' in col.lower():
                text_col = col
                break
                
        if text_col is None:
            st.error("Could not find a column named 'text' or 'review' in your CSV. Please rename your column and try again.")
        else:
            st.info(f"Found review column: **{text_col}**. Processing {len(df)} rows...")
            
            if st.button("Start Batch Processing 🚀"):
                with st.spinner(f"Cleaning and predicting {len(df)} reviews..."):
                    # Clean and Predict
                    df['clean_text'] = df[text_col].astype(str).apply(clean_text)
                    vectors = vectorizer.transform(df['clean_text'])
                    df['AI_Sentiment'] = model.predict(vectors)
                    
                    # Drop the temporary clean text column
                    df = df.drop(columns=['clean_text'])
                    
                    st.success("Batch Processing Complete!")
                    st.dataframe(df.head(10)) # Preview top 10
                    
                    # Create downloadable CSV
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Predicted Results 📥",
                        data=csv,
                        file_name='ai_sentiment_results.csv',
                        mime='text/csv',
                    )

# ================= TAB 3: HISTORY =================
with tab3:
    st.markdown("### Recent Predictions")
    if len(st.session_state.history) == 0:
        st.write("No predictions made yet in this session.")
    else:
        for item in st.session_state.history:
            color = "#22c55e" if item['pred'] == 'Positive' else "#ef4444" if item['pred'] == 'Negative' else "#eab308"
            st.markdown(f"""
            <div class='history-box'>
                <div class='history-text'>"{item['text']}"</div>
                <div class='history-label' style='color: {color};'>Prediction: {item['pred']}</div>
            </div>
            """, unsafe_allow_html=True)
