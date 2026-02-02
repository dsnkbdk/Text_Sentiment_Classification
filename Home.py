import streamlit as st

st.set_page_config(page_title="Crypto News Sentiment Classification", layout="wide")
st.title("Crypto News Sentiment Classification Platform", text_alignment="center")

st.divider()

tool_row = st.columns(5)
with tool_row[0]:
    st.subheader("🪙 Crypto News +", text_alignment="center")
with tool_row[1]:
    st.subheader("🐍 Python 3.12", text_alignment="center")
with tool_row[2]:
    st.subheader("🧪 MLflow", text_alignment="center")
with tool_row[3]:
    st.subheader("⚡ FastAPI", text_alignment="center")
with tool_row[4]:
    st.subheader("📊 Streamlit", text_alignment="center")

st.divider()

st.subheader("🏗️ System Architecture", text_alignment="center")

with st.container(horizontal_alignment="center", vertical_alignment="center"):
    st.image("assets/Architecture.png")

st.markdown(
    """
    Welcome!

    Open pages from the **left sidebar**:

    - 📊 **Dashboard**: Explore sentiment trends and breakdowns.
    - 🌐 **Serving UI**: Call FastAPI to run the model.
    

    """
)