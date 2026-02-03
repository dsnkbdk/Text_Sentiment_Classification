import pandas as pd
import streamlit as st
import plotly.express as px
from data import get_clean_data

st.set_page_config(page_title="Crypto News Sentiment Dashboard", layout="wide")
st.title("Crypto News Sentiment Dashboard", text_alignment="center")

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

COLOR_MAP = {
    "negative": "#1d76e3",
    "neutral":  "#8bcafa",
    "positive": "#fca09d",
}

# Count sentiment labels over time
def sentiment_count_over_time(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    return (
        df.groupby([pd.Grouper(key="date", freq=freq), "class"])
        .size()
        .reset_index(name="count")
    )

# Calculate sentiment share over time
def sentiment_share_over_time(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    group = (
        df.groupby([pd.Grouper(key="date", freq=freq), "class"])
        .size()
        .reset_index(name="count")
    )
    total = group.groupby("date")["count"].transform("sum")
    group["share"] = group["count"] / total
    return group

# Calculate mean polarity score over time
def polarity_mean_over_time(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    return (
        df.groupby(pd.Grouper(key="date", freq=freq))["polarity"]
        .mean()
        .reset_index(name="polarity_mean")
    )

# Calculate mean subjectivity score over time
def subjectivity_mean_over_time(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    return (
        df.groupby(pd.Grouper(key="date", freq=freq))["subjectivity"]
        .mean()
        .reset_index(name="subjectivity_mean")
    )


# Load clean data
df = st.cache_data()(get_clean_data)("oliviervha/crypto-news", "cryptonews.csv")

# Sidebar settings
# Filters
st.sidebar.header("Filters")

# Date range
min_date = df["date"].min().date()
max_date = df["date"].max().date()

if "date_range" not in st.session_state:
    st.session_state["date_range"] = (min_date, max_date)

if st.sidebar.button("Reset date range", width="stretch"):
    st.session_state["date_range"] = (min_date, max_date)
    st.rerun()

date_range = st.sidebar.date_input(
    label="Date range",
    min_value=min_date,
    max_value=max_date,
    key="date_range"
)

# Time granularity
freq = st.sidebar.selectbox(
    label="Time granularity",
    options=["Daily", "Weekly", "Monthly", "Yearly"],
    index=0
)

freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME", "Yearly": "YE"}

# Source & Subject
sources = st.sidebar.multiselect("Source", sorted(df["source"].unique()))
subjects = st.sidebar.multiselect("Subject", sorted(df["subject"].unique()))

# Trends
st.sidebar.header("Trends")

trend_metric = st.sidebar.radio(
    label="Trend metric",
    options=[
        "Share (Stacked)",
        "Count (Stacked)",
        "Polarity mean (Line)",
        "Subjectivity mean (Line)"
    ],
    index=0
)

# Filter out a subset based on date range
start = pd.to_datetime(date_range[0])
end = pd.to_datetime(date_range[1] if len(date_range) == 2 else None) + pd.Timedelta(days=1)

mask = (df["date"] >= start) & (df["date"] < end)

if sources:
    mask &= df["source"].isin(sources)
if subjects:
    mask &= df["subject"].isin(subjects)

sub_df = df.loc[mask].copy()

# Main page settings
# KPI section
st.divider()

row1 = st.columns(3)
row1[0].metric("Rows", f"{len(sub_df)}")
row1[1].metric("Sources", f"{sub_df['source'].nunique()}")
row1[2].metric("Subjects", f"{sub_df['subject'].nunique():,}")

st.divider()

row2 = st.columns(5)
row2[0].metric("Negative share", f"{sub_df['class'].eq('negative').mean():.1%}")
row2[1].metric("Neutral share", f"{sub_df['class'].eq('neutral').mean():.1%}")
row2[2].metric("Positive share", f"{sub_df['class'].eq('positive').mean():.1%}")
row2[3].metric("Avg polarity", f"{sub_df['polarity'].mean():.3f}")
row2[4].metric("Avg subjectivity", f"{sub_df['subjectivity'].mean():.3f}")

st.divider()

# Trend section
st.subheader("Trend over time")

if trend_metric == "Share (Stacked)":
    trend = sentiment_share_over_time(sub_df, freq=freq_map[freq])
    fig_trend = px.area(
        trend,
        x="date",
        y="share",
        color="class",
        color_discrete_map=COLOR_MAP
    )
    fig_trend.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_trend)

elif trend_metric == "Count (Stacked)":
    trend = sentiment_count_over_time(sub_df, freq=freq_map[freq])
    fig_trend = px.area(
        trend,
        x="date",
        y="count",
        color="class",
        color_discrete_map=COLOR_MAP
    )
    st.plotly_chart(fig_trend)

elif trend_metric == "Polarity mean (Line)":
    trend = polarity_mean_over_time(sub_df, freq=freq_map[freq])
    fig_trend = px.line(trend, x="date", y="polarity_mean")
    fig_trend.update_traces(connectgaps=True)
    st.plotly_chart(fig_trend)

elif trend_metric == "Subjectivity mean (Line)":
    trend = subjectivity_mean_over_time(sub_df, freq=freq_map[freq])
    fig_trend = px.line(trend, x="date", y="subjectivity_mean")
    fig_trend.update_traces(connectgaps=True)
    st.plotly_chart(fig_trend)

st.divider()

# Breakdown bar charts
left_source, right_subject = st.columns(2)

with left_source:
    st.subheader("Sentiment distribution by source", text_alignment="center")
    fig_source = px.histogram(
        sub_df,
        x="source",
        barmode="group",
        color="class",
        color_discrete_map=COLOR_MAP
    )
    st.plotly_chart(fig_source)

with right_subject:
    st.subheader("Sentiment distribution by subject", text_alignment="center")
    fig_subject = px.histogram(
        sub_df,
        x="subject",
        barmode="group",
        color="class",
        color_discrete_map=COLOR_MAP
    )
    st.plotly_chart(fig_subject)

st.divider()

st.subheader("Further Analysis")

analysis_mode = st.radio("Analysis by", options=["source", "subject"], horizontal=True)
selected = st.radio(
    label=f"Select a {analysis_mode}",
    options=sorted(sub_df[analysis_mode].unique().tolist()),
    horizontal=True
)

if selected:
    analysis_df = sub_df[sub_df[analysis_mode] == selected].copy()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Rows", f"{len(analysis_df)}")
    c2.metric("Negative share", f"{analysis_df['class'].eq('negative').mean():.1%}")
    c3.metric("Neutral share", f"{analysis_df['class'].eq('neutral').mean():.1%}")
    c4.metric("Positive share", f"{analysis_df['class'].eq('positive').mean():.1%}")
    c5.metric("Avg polarity", f"{analysis_df['polarity'].mean():.3f}")
    c6.metric("Avg subjectivity", f"{analysis_df['subjectivity'].mean():.3f}")

    # Distribution
    dist = analysis_df["class"].value_counts().reset_index()
    dist.columns = ["class", "count"]
    fig = px.bar(dist, x="class", y="count")
    st.plotly_chart(fig)

    st.divider()

    # Latest news samples
    st.subheader("Latest samples")

    sample_n = st.slider("Show the latest N news", 10, 100, 50)
    cols = [c for c in ["date", "source", "subject", "class", "title", "text", "url"] if c in analysis_df.columns]
    st.dataframe(analysis_df.sort_values("date", ascending=False)[cols].head(sample_n))