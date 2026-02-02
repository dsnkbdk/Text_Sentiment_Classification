import os
import json
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
FASTAPI_URI = os.getenv("FASTAPI_URI", "http://localhost:8000")

st.set_page_config(page_title="Model Monitor & Compare", layout="wide")
st.title("Model Monitor & Compare", anchor=False)

# --------
# Helpers
# --------
def _short_text(s: str, n: int = 60) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else (s[:n] + "...")

def _get(url: str, timeout: int = 10) -> tuple[bool, dict | None, str]:
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return False, None, f"{r.status_code}: {r.text}"
        return True, r.json(), ""
    except Exception as e:
        return False, None, str(e)

def _post(url: str, payload: dict, timeout: int = 30) -> tuple[bool, dict | None, str]:
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        if r.status_code != 200:
            return False, None, f"{r.status_code}: {r.text}"
        return True, r.json(), ""
    except requests.exceptions.Timeout:
        return False, None, "Request timeout (LLM may be slow on CPU)."
    except Exception as e:
        return False, None, str(e)

def _extract_ml_row(pred: dict) -> tuple[str | None, float | None]:
    """
    Return (sentiment, max_prob) from ML predict response item.
    """
    sentiment = pred.get("sentiment")
    probs = pred.get("probabilities") or {}
    try:
        max_prob = max((float(v) for v in probs.values()), default=None)
    except Exception:
        max_prob = None
    return sentiment, max_prob

def _extract_llm_row(pred: dict) -> tuple[str | None, float | None]:
    """
    Return (sentiment, score) from LLM predict response item.
    """
    sentiment = pred.get("sentiment")
    score = pred.get("score")
    try:
        score = float(score) if score is not None else None
    except Exception:
        score = None
    return sentiment, score


# -----------------------
# Section 1: Model Monitor
# -----------------------
st.subheader("1) Model Monitor")

c1, c2, c3, c4 = st.columns(4)
if c1.button("Show FASTAPI_URI", width="stretch"):
    st.info(FASTAPI_URI)

if c2.button("Ping /health", width="stretch"):
    ok, data, err = _get(f"{FASTAPI_URI}/health")
    st.success(data) if ok else st.error(err)

if c3.button("Refresh /status", width="stretch"):
    pass  # just triggers rerun

c4.link_button(label="Open API docs", url=f"{FASTAPI_URI}/docs", width="stretch")

ok, status, err = _get(f"{FASTAPI_URI}/status")
if not ok:
    st.error(f"Cannot read /status: {err}")
else:
    # Service line
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("API", "Reachable")
    sc2.metric("MLflow Tracking URI", status.get("mlflow_tracking_uri", "N/A"))

    llm_defaults = status.get("llm_defaults", {}) or {}
    sc3.metric(
        "LLM Defaults",
        f"bs={llm_defaults.get('batch_size', 'N/A')} | max_len={llm_defaults.get('max_length', 'N/A')}",
    )

    st.divider()

    # Two model cards
    ml_uri = status.get("ml_model_uri", "N/A")
    llm_uri = status.get("llm_model_uri", "N/A")
    ml_meta = status.get("ml_meta", {}) or {}
    llm_meta = status.get("llm_meta", {}) or {}

    left, right = st.columns(2)

    with left:
        st.markdown("### 🧠 ML Model")
        a1, a2 = st.columns(2)
        a1.metric("Alias", ml_meta.get("alias", "N/A"))
        a2.metric("Version", ml_meta.get("model_version", "N/A"))

        st.text_input("Registry Name", value=ml_meta.get("registered_model_name", "N/A"), disabled=True)
        st.text_input("Model URI", value=ml_uri, disabled=True)
        st.text_input("Run ID", value=ml_meta.get("run_id", "N/A"), disabled=True)

    with right:
        st.markdown("### 🤖 LLM Model")
        b1, b2 = st.columns(2)
        b1.metric("Alias", llm_meta.get("alias", "N/A"))
        b2.metric("Version", llm_meta.get("model_version", "N/A"))

        st.text_input("Registry Name", value=llm_meta.get("registered_model_name", "N/A"), disabled=True)
        st.text_input("Model URI", value=llm_uri, disabled=True)
        st.text_input("Run ID", value=llm_meta.get("run_id", "N/A"), disabled=True)

    with st.expander("Raw /status JSON"):
        st.json(status)

st.divider()


# ------------------------
# Section 2: Model Compare
# ------------------------
st.subheader("2) Model Compare")

st.caption("Run both models on the same inputs and compare predictions.")

DEFAULT_SAMPLES = [
    "Bitcoin rallies sharply after ETF approval, igniting strong investor enthusiasm.",
    "Ethereum trades sideways as investors await macroeconomic data.",
    "Crypto prices plunge after exchange hack fears spark panic and heavy regulation."
]

mode = st.radio("Input Mode", ["Single", "Batch"], horizontal=True)

if "cmp_single_text" not in st.session_state:
    st.session_state["cmp_single_text"] = DEFAULT_SAMPLES[0]
if "cmp_batch_text" not in st.session_state:
    st.session_state["cmp_batch_text"] = "\n".join(DEFAULT_SAMPLES)

if mode == "Single":
    st.text_area("Text", height=160, key="cmp_single_text")
    texts = [st.session_state["cmp_single_text"].strip()] if st.session_state["cmp_single_text"].strip() else []
else:
    st.text_area("Texts (one per line)", height=160, key="cmp_batch_text")
    texts = [ln.strip() for ln in st.session_state["cmp_batch_text"].splitlines() if ln.strip()]

with st.expander("Compare Request Preview"):
    st.code(json.dumps({"input_texts": texts}, indent=2), language="json")

compare = st.button("Run Compare", type="primary")

if compare:
    if not texts:
        st.warning("Please input at least one text.")
    else:
        payload = {"input_texts": texts}

        # Call ML
        ok_ml, ml_res, err_ml = _post(f"{FASTAPI_URI}/predict_ml", payload, timeout=30)
        if not ok_ml:
            st.error(f"ML call failed: {err_ml}")
            st.stop()

        # Call LLM (allow longer timeout)
        ok_llm, llm_res, err_llm = _post(f"{FASTAPI_URI}/predict_llm", payload, timeout=60)
        if not ok_llm:
            st.error(f"LLM call failed: {err_llm}")
            st.stop()

        ml_preds = (ml_res or {}).get("predictions", []) or []
        llm_preds = (llm_res or {}).get("predictions", []) or []

        # Align rows
        n = min(len(texts), len(ml_preds), len(llm_preds))
        rows = []
        for i in range(n):
            ml_sent, ml_maxprob = _extract_ml_row(ml_preds[i])
            llm_sent, llm_score = _extract_llm_row(llm_preds[i])
            rows.append({
                "text": _short_text(texts[i]),
                "ml_sentiment": ml_sent,
                "ml_max_prob": ml_maxprob,
                "llm_sentiment": llm_sent,
                "llm_score": llm_score,
                "match": (ml_sent == llm_sent) if (ml_sent is not None and llm_sent is not None) else None
            })

        df = pd.DataFrame(rows)

        # KPIs
        k1, k2, k3 = st.columns(3)
        k1.metric("Compared items", n)
        if "match" in df.columns and n > 0:
            match_rate = df["match"].mean() if df["match"].notna().any() else None
            k2.metric("Match rate", f"{match_rate:.1%}" if match_rate is not None else "N/A")
        k3.metric("LLM latency (ms)", (llm_res or {}).get("latency_ms", "N/A"))

        st.dataframe(df, use_container_width=True)

        with st.expander("Raw ML response"):
            st.json(ml_res)
        with st.expander("Raw LLM response"):
            st.json(llm_res)
