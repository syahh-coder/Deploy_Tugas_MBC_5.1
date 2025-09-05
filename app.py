# Streamlit App: Logistic Regression + TF-IDF (path hidden)
# ----------------------------------------------------------

import re
import io
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# -----------------------------
# Page & Theme
# -----------------------------
st.set_page_config(
    page_title="LR Text Classifier",
    page_icon="🤖",
    layout="wide",
)

# -----------------------------
# Label map
# -----------------------------
LABEL_MAP = {
    0: "Sadness",
    1: "Anger",
    2: "Support",
    3: "Hope",
    4: "Disappointment",
}

CLASS_COLORS = {
    "Sadness": "#636EFA",
    "Anger": "#EF553B",
    "Support": "#00CC96",
    "Hope": "#AB63FA",
    "Disappointment": "#FFA15A",
}

# -----------------------------
# Settings (hardcoded paths)
# -----------------------------

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model_LR.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
PROB_DECIMALS = 3
CONF_THRESHOLD = 0.0

# -----------------------------
# Utilities
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_model_and_vectorizer():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

def clean_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text

def predict_proba(texts: List[str], model, vectorizer) -> Tuple[np.ndarray, List[int]]:
    X = vectorizer.transform(texts)
    probs = model.predict_proba(X)
    preds = probs.argmax(axis=-1).tolist()
    return probs, preds

# -----------------------------
# Header
# -----------------------------
st.title("🤖 NLP Multi-class Classifier (LR + TF-IDF)")
st.markdown(
    "Masukkan teks Bahasa Indonesia, model akan mengklasifikasikan ke 5 kelas: "
    "**Sadness, Anger, Support, Hope, Disappointment**."
)

# Load model & vectorizer
with st.spinner("Loading model & TF-IDF vectorizer…"):
    try:
        model, vectorizer = load_model_and_vectorizer()
    except Exception as e:
        st.error("Gagal memuat model/vectorizer. Pastikan file ada di path yang benar.")
        st.exception(e)
        st.stop()

# Tabs: Single / Batch
tab_single, tab_batch = st.tabs(["🔤 Single Text", "📄 Batch CSV"])

# -----------------------------
# Single Text Tab
# -----------------------------
with tab_single:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        example_texts = [
            "Saya sangat sedih karena rencana gagal total.",
            "Kenapa sih pelayanan seperti ini! Bikin marah saja.",
            "Tetap semangat ya, kami dukung kamu!",
            "Aku optimis, besok pasti lebih baik.",
            "Jujur kecewa dengan hasilnya, tidak sesuai harapan.",
        ]
        ex_choice = st.selectbox("Contoh cepat", options=["(none)"] + example_texts, index=0)
        default_text = "" if ex_choice == "(none)" else ex_choice
        user_text = st.text_area("Masukkan teks", value=default_text, height=150)
        run_btn = st.button("🚀 Klasifikasikan")

    with col_right:
        st.subheader("🔧 Preprocessing")
        st.write("- Lowercase\n- Trim spasi\n- Hapus punctuation")
        st.caption("Tidak ada stemming/stopword removal.")

    if run_btn and user_text.strip():
        cleaned = clean_text(user_text)
        probs, preds = predict_proba([cleaned], model, vectorizer)
        probs = probs[0]
        pred_idx = int(preds[0])
        pred_label = LABEL_MAP.get(pred_idx)
        top_prob = float(probs[pred_idx])

        badge_color = CLASS_COLORS.get(pred_label, "#1f77b4")
        st.markdown(
            f"<div style='padding:10px;border-radius:12px;background:{badge_color};color:white;display:inline-block;'>"
            f"Prediksi: <b>{pred_label}</b> • Prob: {top_prob:.{PROB_DECIMALS}f}"
            f"</div>", unsafe_allow_html=True
        )

        if CONF_THRESHOLD > 0 and top_prob < CONF_THRESHOLD:
            st.info("⚠️ Confidence rendah.")

        df_probs = pd.DataFrame({
            "Class": [LABEL_MAP[i] for i in range(len(probs))],
            "Probability": probs,
        })
        fig = px.bar(df_probs, x="Class", y="Probability",
                     title="Class Probabilities", range_y=[0, 1],
                     text=[f"{p:.{PROB_DECIMALS}f}" for p in probs])
        fig.update_traces(textposition="outside")
        fig.update_traces(marker_color=[CLASS_COLORS.get(c, None) for c in df_probs["Class"]])
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Batch Tab
# -----------------------------
with tab_batch:
    st.write("Upload CSV dengan kolom **text** → aplikasi menambahkan kolom pred_label dan prob_<class>.")
    up = st.file_uploader("Upload CSV", type=["csv"])

    if up is not None:
        df = pd.read_csv(up)
        if "text" not in df.columns:
            st.error("CSV harus ada kolom 'text'.")
        else:
            with st.spinner("Predicting…"):
                texts = df["text"].astype(str).apply(clean_text).tolist()
                batch_size = 64
                all_probs = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    probs, preds = predict_proba(batch, model, vectorizer)
                    all_probs.append(probs)
                probs = np.vstack(all_probs)
                preds = probs.argmax(axis=-1)

            df["pred_label"] = [LABEL_MAP.get(int(i)) for i in preds]
            for i in range(probs.shape[1]):
                df[f"prob_{LABEL_MAP.get(i)}"] = probs[:, i]

            st.success("Selesai memproses.")
            st.dataframe(df.head(20), use_container_width=True)

            dist = df["pred_label"].value_counts().reset_index()
            dist.columns = ["Class", "Count"]
            fig2 = px.bar(dist, x="Class", y="Count", title="Prediction Distribution")
            fig2.update_traces(marker_color=[CLASS_COLORS.get(c, None) for c in dist["Class"]])
            st.plotly_chart(fig2, use_container_width=True)

            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=False)
            st.download_button("💾 Download Predictions", data=csv_buf.getvalue(),
                               file_name="predictions.csv", mime="text/csv")
