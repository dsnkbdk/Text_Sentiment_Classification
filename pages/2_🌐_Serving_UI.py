import os
import json
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
FASTAPI_URI=os.getenv("FASTAPI_URI", "http://localhost:8000")

SAMPLES = [
    "Bitcoin rallies sharply after ETF approval, igniting strong investor enthusiasm.",
    "Ethereum trades sideways as investors await macroeconomic data.",
    "Crypto prices plunge after exchange hack fears spark panic and heavy regulation."
]

MODEL_OPTIONS = {
    "ML (TFIDF_Logistic_Regression)": {
        "endpoint": "/predict_ml",
        "type": "ml"
    },
    "LLM (HF_Cardiffnlp_RoBERTa_Sentiment)": {
        "endpoint": "/predict_llm",
        "type": "llm"
    }
}

st.set_page_config(page_title="Sentiment Classification Model Serving UI", layout="wide")
st.title("Sentiment Classification Model Serving UI", text_alignment="center")

# Global settings
st.markdown(
    """
    <style>
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

# Connection
st.subheader(f"API connection")

def ping(url: str) -> tuple[bool, str]:
    try:
        r = requests.get(url, timeout=10)
        return True, f"{r.status_code}: {r.text}"
    except Exception as e:
        return False, str(e)

# Button
b1, b2, b3, b4 = st.columns(4)

# Define button behavior
if b1.button("API URL", width="stretch"):
    st.info(FASTAPI_URI)

if b2.button("Ping /health", width="stretch"):
    ok, msg = ping(f"{FASTAPI_URI}/health")
    _ = st.success(msg) if ok else st.error(msg)

if b3.button("Ping /status", width="stretch"):
    ok, msg = ping(f"{FASTAPI_URI}/status")
    _ = st.success(msg) if ok else st.error(msg)

b4.link_button(label="Open API docs", url=f"{FASTAPI_URI}/docs", width="stretch")

st.divider()

# Prediction
st.subheader("Sentiment Prediction")

# Model selection
model_name = st.radio(
    "Model",
    list(MODEL_OPTIONS.keys()),
    horizontal=True
)
model_cfg = MODEL_OPTIONS[model_name]
endpoint = model_cfg["endpoint"]
model_type = model_cfg["type"]

st.write(f"Calling endpoint: {FASTAPI_URI}{endpoint}")

# Support both single and batch
mode = st.radio("Mode", ["Single Test", "Batch Test"], horizontal=True)

# Session state
if "single_text" not in st.session_state:
    st.session_state["single_text"] = ""
if "batch_text" not in st.session_state:
    st.session_state["batch_text"] = ""

# Quick samples
if mode == "Single Test":
    st.write("Single example")
    
    # Define button behavior
    c1, c2, c3 = st.columns(3)
    if c1.button("Positive sample", width="stretch"):
        st.session_state["single_text"] = SAMPLES[0]
    if c2.button("Neutral sample", width="stretch"):
        st.session_state["single_text"] = SAMPLES[1]
    if c3.button("Negative sample", width="stretch"):
        st.session_state["single_text"] = SAMPLES[2]

    # Input area
    single_text = st.text_area(
        "Single text",
        height=200,
        placeholder="Paste a crypto news here...",
        key="single_text"
    )
    payload = {"input_text": single_text.strip()}

else:
    st.write("Batch example")

    # Define button behavior
    if st.button("Batch example", width="stretch"):
        st.session_state["batch_text"] = "\n".join(SAMPLES)

    # Input area
    batch_text = st.text_area(
        "Batch texts",
        height=200,
        placeholder="Paste a crypto news here...",
        key="batch_text"
    )
    payload = {"input_texts": [ln.strip() for ln in batch_text.splitlines() if ln.strip()]}

# Preview
with st.expander("Request JSON preview"):
    st.code(json.dumps(payload, indent=2), language="json")

def clear_input():
    st.session_state["single_text"] = ""
    st.session_state["batch_text"] = ""

# Button
left_button, right_button = st.columns(2)
clear = right_button.button("Clear", on_click=clear_input, width="stretch")
predict = left_button.button("Predict", type="primary", width="stretch")

# Define button behavior
if predict:

    if mode == "Single Test" and not payload["input_text"]:
        st.warning("Please enter some text")
    elif mode == "Batch Test" and not payload["input_texts"]:
        st.warning("Please enter some text")
    else:
        try:
            r = requests.post(url=f"{FASTAPI_URI}{endpoint}", json=payload, timeout=20)
            
            if r.status_code != 200:
                st.error(f"API error: {r.status_code}")
                st.code(r.text)
            else:
                data = r.json()

                # Metadata
                st.subheader("Model Info")
                meta = data.get("meta", {})
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Latency (ms)", data.get("latency_ms", "N/A"))
                c2.metric("Model Alias", meta.get("alias", "N/A"))
                c3.metric("Model Version", meta.get("model_version", "N/A"))
                
                with st.expander("Full metadata"):
                    st.json(meta)

                # Predictions
                st.subheader("Predictions")
                preds = data.get("predictions", [])

                rows = []

                # Single
                if mode == "Single Test":
                    pred = preds[0]
                    # ML
                    if model_type == "ml":
                        probs = pred.get("probabilities") or {}
                    
                        rows.append({
                            "text": (payload["input_text"][:30] + "..."),
                            "sentiment": pred.get("sentiment"),
                            "prob_negative": probs.get("negative"),
                            "prob_neutral": probs.get("neutral"),
                            "prob_positive": probs.get("positive")
                        })
                    # LLM
                    else:
                        rows.append({
                            "text": (payload["input_text"][:30] + "..."),
                            "sentiment": pred.get("sentiment"),
                            "score": pred.get("score")
                        })
                # Batch
                else:
                    for i, pred in enumerate(preds):
                        # ML
                        if model_type == "ml":
                            probs = pred.get("probabilities") or {}
                            rows.append({
                                "text": (payload["input_texts"][i][:30] + "..."),
                                "sentiment": pred.get("sentiment"),
                                "prob_negative": probs.get("negative"),
                                "prob_neutral": probs.get("neutral"),
                                "prob_positive": probs.get("positive")
                            })
                        # LLM
                        else:
                            rows.append({
                                "text": (payload["input_texts"][i][:30] + "..."),
                                "sentiment": pred.get("sentiment"),
                                "score": pred.get("score")
                            })
                
                # Unified output
                if rows:
                    st.dataframe(pd.DataFrame(rows))

                # Raw response
                with st.expander("Raw API Response"):
                    st.json(data)
        
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API")
        except requests.exceptions.Timeout:
            st.error("Request timeout, try again")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
