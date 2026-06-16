import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import urllib.request
import tensorflow as tf
from PyPDF2 import PdfReader
from docx import Document
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="AI vs Human Text Detector", layout="wide")
st.title("🔍 AI vs. Human Text Detection Dashboard")
st.markdown("Upload a document or paste raw text to run evaluation across multiple trained classifiers.")

# --- Document Parsing Helpers ---
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# --- Text Statistics Helper ---
def extract_linguistic_features(text):
    words = text.split()
    sentences = [s for s in text.split('.') if len(s.strip()) > 0]
    avg_word_len = sum(len(w) for w in words) / (len(words) + 1e-5)
    avg_sent_len = len(words) / (len(sentences) + 1e-5)
    vocab_richness = len(set(words)) / (len(words) + 1e-5)
    return avg_word_len, avg_sent_len, vocab_richness

# --- Dynamic Model Asset Manager ---
@st.cache_resource
def load_all_models():
    # Ensure our local models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Base URL pointing directly to your v1.0.0 release assets
    base_release_url = "https://github.com/LeoLakshman/ai_human_detection_project/releases/download/v1.0.0"
    
    # List of all 8 files we need to pull down if they aren't cached locally
    model_files = [
        "tfidf_vectorizer.pkl",
        "svm_model.pkl",
        "decision_tree_model.pkl",
        "adaboost_model.pkl",
        "dl_tokenizer.pkl",
        "fnn_model.h5",
        "lstm_model.h5",
        "cnn_model.h5"
    ]
    
    # Create a clean loading status indicator in the UI so the user knows it's working
    with st.spinner("Downloading and initializing model artifacts from secure cloud release..."):
        for file in model_files:
            local_path = os.path.join('models', file)
            if not os.path.exists(local_path):
                download_url = f"{base_release_url}/{file}"
                try:
                    urllib.request.urlretrieve(download_url, local_path)
                except Exception as e:
                    st.error(f"Failed to fetch {file} from release. Error context: {e}")
                    st.stop()

    # Now that we've verified all files exist locally, we can read them into memory
    tfidf = joblib.load('models/tfidf_vectorizer.pkl')
    svm = joblib.load('models/svm_model.pkl')
    if hasattr(svm, "probability") and svm.probability:
        svm._effective_probability = True
    dt = joblib.load('models/decision_tree_model.pkl')
    ada = joblib.load('models/adaboost_model.pkl')
    tokenizer = joblib.load('models/dl_tokenizer.pkl')
    
    fnn = tf.keras.models.load_model('models/fnn_model.h5')
    lstm = tf.keras.models.load_model('models/lstm_model.h5')
    cnn = tf.keras.models.load_model('models/cnn_model.h5')
    
    return tfidf, svm, dt, ada, tokenizer, fnn, lstm, cnn

# Trigger the cached loader function
tfidf, svm, dt, ada, tokenizer, fnn, lstm, cnn = load_all_models()

# --- Layout Configuration ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Input Content")
    doc_file = st.file_uploader("Upload Document (.pdf, .docx)", type=["pdf", "docx"])
    typed_text = st.text_area("Or paste your text block directly here:", height=250)
    
    final_text = ""
    if doc_file is not None:
        if doc_file.name.endswith('.pdf'):
            final_text = extract_text_from_pdf(doc_file)
        elif doc_file.name.endswith('.docx'):
            final_text = extract_text_from_docx(doc_file)
    elif typed_text.strip() != "":
        final_text = typed_text

    model_choice = st.selectbox(
        "Select Active Classifier Matrix",
        ["SVM", "Decision Tree", "AdaBoost", "Feedforward NN (FNN)", "LSTM", "1D CNN"]
    )

with col2:
    st.subheader("📊 Evaluation Metrics & Breakdown")
    if final_text.strip() == "":
        st.info("Awaiting input data sequence. Paste text or upload a document file to start model inference.")
    else:
        # Generate and show running structural statistics
        w_len, s_len, v_rich = extract_linguistic_features(final_text)
        
        st.markdown(f"**Word Count:** {len(final_text.split())} words")
        st.markdown(f"**Average Word Length:** {w_len:.2f} characters")
        st.markdown(f"**Average Sentence Length:** {s_len:.2f} words")
        st.markdown(f"**Vocabulary Richness (TTR):** {v_rich:.2f}")
        
        st.write("---")
        
        # Vectorize text strings for traditional machine learning inputs
        X_ml = tfidf.transform([final_text])
        
        # Tokenize and pad input matrices for spatial deep learning inputs
        seq = tokenizer.texts_to_sequences([final_text])
        X_dl = pad_sequences(seq, maxlen=200, padding='post')
        
        prediction, confidence = 0, 0.0
        
        # Selection routing logic block
        if model_choice == "SVM":
            prediction = svm.predict(X_ml)[0]
            confidence = svm.predict_proba(X_ml)[0][prediction] * 100
        elif model_choice == "Decision Tree":
            prediction = dt.predict(X_ml)[0]
            confidence = dt.predict_proba(X_ml)[0][prediction] * 100
        elif model_choice == "AdaBoost":
            prediction = ada.predict(X_ml)[0]
            confidence = ada.predict_proba(X_ml)[0][prediction] * 100
        elif model_choice == "Feedforward NN (FNN)":
            prob = fnn.predict(X_ml.toarray(), verbose=0)[0][0]
            prediction = 1 if prob > 0.5 else 0
            confidence = prob * 100 if prediction == 1 else (1 - prob) * 100
        elif model_choice == "LSTM":
            prob = lstm.predict(X_dl, verbose=0)[0][0]
            prediction = 1 if prob > 0.5 else 0
            confidence = prob * 100 if prediction == 1 else (1 - prob) * 100
        elif model_choice == "1D CNN":
            prob = cnn.predict(X_dl, verbose=0)[0][0]
            prediction = 1 if prob > 0.5 else 0
            confidence = prob * 100 if prediction == 1 else (1 - prob) * 100

        # --- Display Final Output Metrics ---
        if prediction == 1:
            st.error(f"🚨 **AI-Generated Text Detected**")
            st.metric(label="Model Confidence Score", value=f"{confidence:.2f}%")
        else:
            st.success(f"✍️ **Human-Written Text Verified**")
            st.metric(label="Model Confidence Score", value=f"{confidence:.2f}%")
