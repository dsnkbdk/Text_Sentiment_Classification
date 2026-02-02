import os
import json
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import plotly.graph_objects as go

load_dotenv()
FASTAPI_URI=os.getenv("FASTAPI_URI", "http://localhost:8000")

SAMPLES = [
    "Bitcoin rallies sharply after ETF approval, igniting strong investor enthusiasm.",
    "Ethereum trades sideways as investors await macroeconomic data.",
    "Crypto prices plunge after exchange hack fears spark panic and heavy regulation."
]

st.set_page_config(page_title="Model Monitor", layout="wide")
st.title("Model Monitor", text_alignment="center")

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
    /* success or error */
    div[data-testid="stAlert"] {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def ping(url: str, timeout: int = 10) -> tuple[bool, object]:
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return False, f"{r.status_code}: {r.text}"
        return True, r.json()
    except Exception as e:
        return False, str(e)


def post(url: str, payload: dict, timeout: int = 20) -> tuple[bool, object]:
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        if r.status_code != 200:
            return False, f"{r.status_code}: {r.text}"
        return True, r.json()
    except requests.exceptions.Timeout:
        return False, "Request timeout, try again"
    except Exception as e:
        return False, str(e)


def extract_ml_row(pred: dict) -> tuple[str | None, float | None]:
    sentiment = pred.get("sentiment")
    probs = pred.get("probabilities") or {}
    try:
        max_prob = max((v for v in probs.values()), default=None)
    except Exception:
        max_prob = None
    return sentiment, max_prob


def extract_llm_row(pred: dict) -> tuple[str | None, float | None]:
    sentiment = pred.get("sentiment")
    score = pred.get("score")
    try:
        score = score if score is not None else None
    except Exception:
        score = None
    return sentiment, score


def light(label: str, state: bool | None, ok_text: str, err_text: str):
    if state is None:
        st.warning(f"{label}: Not checked", icon="🟡")
    elif state:
        st.success(ok_text, icon="🟢")
    else:
        st.error(err_text, icon="🔴")


def gauge_latency(title: str, value_ms: int, max_ms: int) -> go.Figure:
    v = int(value_ms) if value_ms is not None else 0

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=v,
            title={"text": title},
            number={"suffix": " ms"},
            gauge={
                "axis": {"range": [0, max_ms]},
                "bar": {"color": "gray"}
            }
        )
    )
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=60, b=20))
    
    return fig

# Empty latency gauges
if "ml_latency_ms" not in st.session_state:
    st.session_state["ml_latency_ms"] = 0
if "llm_latency_ms" not in st.session_state:
    st.session_state["llm_latency_ms"] = 0

# Model status
st.subheader("Model Status")

# Set button
model_status = st.button("Check Model Status", width="stretch")

# Status init
ok_health: bool | None = None
ok_status: bool | None = None
msg_status: dict = {}

if model_status:
    ok_health, _ = ping(f"{FASTAPI_URI}/health")
    ok_status, msg_status = ping(f"{FASTAPI_URI}/status")

# Traffic light
c1, c2, c3 = st.columns(3)

with c1:
    light(
        "API Health",
        ok_health,
        ok_text=f"API Health: {FASTAPI_URI}/health",
        err_text=f"API Health: {FASTAPI_URI}/health"
    )
with c2:
    light(
        "API Status",
        ok_status,
        ok_text=f"API Status: {FASTAPI_URI}/status",
        err_text=f"API Status: {FASTAPI_URI}/status"
    )
with c3:
    if ok_status is None:
        st.warning(f"Mlflow URI: Not checked", icon="🟡")
    elif ok_status:
        mlflow_tracking_uri = msg_status.get("mlflow_tracking_uri")
        
        if mlflow_tracking_uri:
            st.success(f"Mlflow URI: {mlflow_tracking_uri}", icon="🟢")
        else:
            st.error("Mlflow URI: N/A", icon="🔴")
    else:
        st.error("Mlflow URI: N/A", icon="🔴")


st.divider()


# Model info placeholders
if ok_status:
    ml_meta = msg_status.get("ml_meta", {}) or {}
    llm_meta = msg_status.get("llm_meta", {}) or {}

    # ML
    ml_alias = ml_meta.get("alias", "N/A")
    ml_version = ml_meta.get("model_version", "N/A")

    ml_registry_name = ml_meta.get("registered_model_name", "N/A")
    ml_model_uri = msg_status.get("ml_model_uri", "N/A")
    ml_run_id = ml_meta.get("run_id", "N/A")

    # LLM
    llm_alias = llm_meta.get("alias", "N/A")
    llm_version = llm_meta.get("model_version", "N/A")

    llm_registry_name = llm_meta.get("registered_model_name", "N/A")
    llm_model_uri = msg_status.get("llm_model_uri", "N/A")
    llm_run_id = llm_meta.get("run_id", "N/A")
else:
    # Before click placeholder
    ml_alias = "N/A"
    ml_version = "N/A"
    ml_registry_name = "N/A"
    ml_model_uri = "N/A"
    ml_run_id = "N/A"

    llm_alias = "N/A"
    llm_version = "N/A"
    llm_registry_name = "N/A"
    llm_model_uri = "N/A"
    llm_run_id = "N/A"

# Info cards
ml_col, llm_col = st.columns(2)

with ml_col:
    st.subheader("Traditional ML", text_alignment="center")
    
    a1, v1 = st.columns(2)
    a1.metric("Alias", ml_alias)
    v1.metric("Version", ml_version)

    st.caption("Registry Name")
    st.code(ml_registry_name, language="markdown")

    st.caption("Model URI")
    st.code(ml_model_uri, language="markdown")

    st.caption("Run ID")
    st.code(ml_run_id, language="markdown")

with llm_col:
    st.subheader("Open-source LLM", text_alignment="center")

    a2, v2 = st.columns(2)
    a2.metric("Alias", llm_alias)
    v2.metric("Version", llm_version)

    st.caption("Registry Name")
    st.code(llm_registry_name, language="markdown")

    st.caption("Model URI")
    st.code(llm_model_uri, language="markdown")

    st.caption("Run ID")
    st.code(llm_run_id, language="markdown")


st.divider()


# Model Compare
st.subheader("Model Compare", help="Run both models on the same inputs and compare outcomes")

# Session state
if "cmp_mode" not in st.session_state:
    st.session_state["cmp_mode"] = None
if "cmp_single_text" not in st.session_state:
    st.session_state["cmp_single_text"] = ""
if "cmp_batch_text" not in st.session_state:
    st.session_state["cmp_batch_text"] = ""

# Result session state
for k, v in {
    "cmp_df": None,
    "cmp_n": 0,
    "cmp_match_rate": None,
    "cmp_raw_ml": None,
    "cmp_raw_llm": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

def set_single():
    st.session_state["cmp_mode"] = "single"
    st.session_state["cmp_single_text"] = SAMPLES[0]

def set_batch():
    st.session_state["cmp_mode"] = "batch"
    st.session_state["cmp_batch_text"] = "\n".join(SAMPLES)

def clear_all():
    # Clear inputs
    st.session_state["cmp_mode"] = None
    st.session_state["cmp_single_text"] = ""
    st.session_state["cmp_batch_text"] = ""

    # Clear compare results
    st.session_state["cmp_df"] = None
    st.session_state["cmp_n"] = 0
    st.session_state["cmp_match_rate"] = None
    st.session_state["cmp_raw_ml"] = None
    st.session_state["cmp_raw_llm"] = None

    # Reset latency dashboard
    st.session_state["ml_latency_ms"] = 0
    st.session_state["llm_latency_ms"] = 0

# Mode buttons
b1, b2 = st.columns(2)
b1.button(
    "Single Test",
    on_click=set_single,
    type="primary" if st.session_state["cmp_mode"] == "single" else "secondary",
    width="stretch",
)
b2.button(
    "Batch Test",
    on_click=set_batch,
    type="primary" if st.session_state["cmp_mode"] == "batch" else "secondary",
    width="stretch",
)

# Inputs
input_texts: list[str] = []

if st.session_state["cmp_mode"] == "single":
    st.text_area(
        "Single text",
        height=200,
        key="cmp_single_text",
        placeholder="Paste a crypto news here..."
    )
    raw_texts = [st.session_state["cmp_single_text"]]
    input_texts = [t.strip() for t in raw_texts if t and t.strip()]

elif st.session_state["cmp_mode"] == "batch":
    st.text_area(
        "Batch texts",
        height=200,
        key="cmp_batch_text",
        placeholder="Paste a crypto news here..."
    )
    raw_texts = st.session_state["cmp_batch_text"].splitlines()
    input_texts = [t.strip() for t in raw_texts if t and t.strip()]

else:
    st.info("Choose a mode above to auto-fill samples")

# Preview
with st.expander("Request JSON preview"):
    st.code(json.dumps({"input_texts": input_texts}, indent=2), language="json")

def clear_input():
    st.session_state["cmp_single_text"] = ""
    st.session_state["cmp_batch_text"] = ""

# Button
left_button, right_button = st.columns(2)
compare = left_button.button("Compare", type="primary", width="stretch")
clear = right_button.button("Clear", on_click=clear_all, width="stretch")

# Define button behavior
if compare:
    if not input_texts:
        st.warning("Please enter some text")
    else:
        payload = {"input_texts": input_texts}
        
        # Call ML
        ok_ml, msg_ml = post(f"{FASTAPI_URI}/predict_ml", payload, timeout=10)
        if not ok_ml:
            st.error(f"ML API error: {msg_ml}")
            st.stop()

        # Call LLM
        ok_llm, msg_llm = post(f"{FASTAPI_URI}/predict_llm", payload, timeout=20)
        if not ok_llm:
            st.error(f"LLM API error: {msg_llm}")
            st.stop()
        
        # Update latency gauges
        st.session_state["ml_latency_ms"] = msg_ml.get("latency_ms") or 0
        st.session_state["llm_latency_ms"] = msg_llm.get("latency_ms") or 0

        ml_preds = (msg_ml or {}).get("predictions", []) or []
        llm_preds = (msg_llm or {}).get("predictions", []) or []

        # Align rows
        n = min(len(input_texts), len(ml_preds), len(llm_preds))

        rows = []
        for i in range(n):
            ml_sentiment, ml_max_prob = extract_ml_row(ml_preds[i])
            llm_sentiment, llm_score = extract_llm_row(llm_preds[i])

            rows.append({
                "text": input_texts[i].strip()[:30] + "...",
                "ml_sentiment": ml_sentiment,
                "ml_max_prob": ml_max_prob,
                "llm_sentiment": llm_sentiment,
                "llm_score": llm_score,
                "match": (ml_sentiment == llm_sentiment)
                if (ml_sentiment is not None and llm_sentiment is not None)
                else None
            })

        df = pd.DataFrame(rows)
        
        match_rate = None
        if "match" in df.columns and n > 0:
            match_rate = df["match"].mean() if df["match"].notna().any() else None

        # Save results for reruns / clear
        st.session_state["cmp_df"] = df
        st.session_state["cmp_n"] = n
        st.session_state["cmp_match_rate"] = match_rate
        st.session_state["cmp_raw_ml"] = msg_ml
        st.session_state["cmp_raw_llm"] = msg_llm

# KPIs
if st.session_state["cmp_df"] is not None:
    k1, k2 = st.columns(2)
    k1.metric("Compared texts", st.session_state["cmp_n"])

    mr = st.session_state["cmp_match_rate"]
    k2.metric("Match rate", f"{mr:.1%}" if mr is not None else "N/A")

    st.dataframe(st.session_state["cmp_df"])

    with st.expander("Raw ML response"):
        st.json(st.session_state["cmp_raw_ml"])

    with st.expander("Raw LLM response"):
        st.json(st.session_state["cmp_raw_llm"])

# Latency Dashboard
st.subheader("Latency Dashboard")

g1, g2 = st.columns(2)
with g1:
    st.plotly_chart(
        gauge_latency(
            "ML latency",
            st.session_state["ml_latency_ms"],
            max_ms=20
        )
    )
with g2:
    st.plotly_chart(
        gauge_latency(
            "LLM latency",
            st.session_state["llm_latency_ms"],
            max_ms=2000
        )
    )

