import os
import json
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
FASTAPI_URI=os.getenv("FASTAPI_URI", "http://localhost:8000")

st.title("Sentiment Classification Model Serving UI", text_alignment="center")

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

# Sample input
if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""

# Sample button
st.write("Quick samples")

c1, c2, c3 = st.columns(3)

# Define button behavior
if c1.button("Positive sample", width="stretch"):
    st.session_state["input_text"] = "Bitcoin price surges after ETF approval and market optimism rises."

if c2.button("Neutral sample", width="stretch"):
    st.session_state["input_text"] = "Ethereum trades sideways as investors await macroeconomic data."

if c3.button("Negative sample", width="stretch"):
    st.session_state["input_text"] = "Crypto market tumbles amid exchange hack fears and regulatory crackdown."

# Input text
text = st.text_area(
    "Input text",
    height=200,
    placeholder="Paste a crypto news title/text here...",
    key="input_text"
)

payload = {"text": text.strip()}

# Preview
with st.expander("Request JSON preview"):
    st.code(json.dumps(payload, indent=2), language="json")

def clear_input():
    st.session_state["input_text"] = ""

# Button
left_button, right_button = st.columns(2)
clear = right_button.button("Clear", on_click=clear_input, width="stretch")
predict = left_button.button("Predict", type="primary", width="stretch")

# Define button behavior
if predict:
    if not text.strip():
        st.warning("Please enter some text")
    else:
        try:
            r = requests.post(url=f"{FASTAPI_URI}/predict", json=payload, timeout=10)
            
            if r.status_code != 200:
                st.error(f"API error: {r.status_code}")
                st.code(r.text)
            else:
                data = r.json()
                st.subheader("Result")
                st.json(data)
        
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
