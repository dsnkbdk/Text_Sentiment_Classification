import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Crypto News Sentiment Classification", layout="wide")
st.title("Crypto News Sentiment Classification Platform", text_alignment="center")

# Global settings
st.markdown(
    """
    <style>
    /* metric */
    [data-testid="stMetricLabel"] {
        display: block;
        text-align: center;
    }
    [data-testid="stMetricValue"] {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.divider()

tool_row = st.columns(5)
with tool_row[0]:
    st.metric("Data", "🪙 Crypto News")
with tool_row[1]:
    st.metric("Language", "🐍 Python 3.12")
with tool_row[2]:
    st.metric("MLOps", "🧪 MLflow")
with tool_row[3]:
    st.metric("Hosting", "⚡ FastAPI")
with tool_row[4]:
    st.metric("Visualisation", "📊 Streamlit")

st.divider()

model_row = st.columns(2)
with model_row[0]:
    st.metric("Traditional ML", "🧠 TF-IDF + Logistic Regression")
with model_row[1]:
    st.metric("Open-source LLM", "🤗 Hugging Face Pipelines")

st.divider()

st.subheader("🏗️ System Architecture", text_alignment="center")

with st.container(horizontal_alignment="center", vertical_alignment="center"):
    st.image("assets/Architecture.png")

st.divider()

st.subheader("Welcome!")

st.markdown(
    """
    Open pages from the **Left Sidebar**
    
    - 📊 **Dashboard:** Explore crypto news sentiment analytics
        - Filter by **Date range**, **Time granularity** (Daily / Weekly / Monthly / Yearly), **Source**, and **Subject**
        - View sentiment **Trend over time** (Share / Count / Polarity mean / Subjectivity mean)
        - Break down sentiment distribution by **Source** and **Subject**
        - Drill down into a selected **Source / Subject** and inspect the **Latest news samples**
    
    <br>
    
    - 🌐 **Serving UI:** Call **FastAPI** to run inference and inspect responses
        - Check API connection: **/health**, **/status**, and open **/docs**
        - Choose model endpoint: **ML (TF-IDF + Logistic Regression)** or **LLM (HF Pipeline + RoBERTa Sentiment)**
        - Support **Single Test** and **Batch Test**, with JSON preview
        - View **Latency**, **Metadata**, **Predictions**, and **Raw Response**

    <br>
        
    - 🧪 **Model Monitor:** Monitor model status and compare ML vs LLM performance
        - One-click status check: API **/health**, **/status**, and **MLflow tracking URI**
        - Show model registry info: **Alias / Version / Model URI / Run ID** for both models
        - Run **Model Compare** on the same inputs and compute **Match rate**
        - Visualise latency with gauges for **ML Latency** vs **LLM Latency**
    """,
    unsafe_allow_html=True
)