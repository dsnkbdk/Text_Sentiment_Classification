import pandas as pd
import streamlit as st
import plotly.express as px
from data import get_clean_data

st.title("Crypto News Sentiment Dashboard")

@st.cache_data

# Count sentiment labels over time
def sentiment_count_over_time(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    group = (
        df.groupby([pd.Grouper(key="date", freq=freq), "class"])
        .size()
        .reset_index(name="count")
    )
    return group

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
    group = (
        df.groupby(pd.Grouper(key="date", freq=freq))["polarity"]
        .mean()
        .reset_index(name="polarity_mean")
    )
    return group

# Calculate mean subjectivity score over time
def subjectivity_mean_over_time(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    group = (
        df.groupby(pd.Grouper(key="date", freq=freq))["subjectivity"]
        .mean()
        .reset_index(name="subjectivity_mean")
    )
    return group


# Load clean data
df = get_clean_data("oliviervha/crypto-news", "cryptonews.csv")

# Sidebar filters
st.sidebar.header("Filters")

min_date = df["date"].min().date()
max_date = df["date"].max().date()

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

freq = st.sidebar.selectbox(
    "Time granularity",
    options=["Daily", "Weekly", "Monthly", "Yearly"],
    index=1
)

freq_map = {
    "Daily": "D",
    "Weekly": "W",
    "Monthly": "ME",
    "Yearly": "YE"
}

freq_code = freq_map[freq]

sources = st.sidebar.multiselect("Source", sorted(df["source"].unique()))
subjects = st.sidebar.multiselect("Subject", sorted(df["subject"].unique()))

st.sidebar.header("Trend settings")

trend_metric = st.sidebar.radio(
    "Trend metric",
    options=["Share (stacked)", "Count (stacked)", "Polarity mean (line)"],
    index=0
)



start = pd.to_datetime(date_range[0])
end = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)

mask = (df["date"] >= start) & (df["date"] < end)
if sources:
    mask &= df["source"].isin(sources)
if subjects:
    mask &= df["subject"].isin(subjects)

fdf = df.loc[mask].copy()

# KPI row
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows", f"{len(fdf):,}")
c2.metric("Sources", f"{fdf['source'].nunique():,}")
c3.metric("Subjects", f"{fdf['subject'].nunique():,}")
c4.metric("Avg polarity", f"{pd.to_numeric(fdf['polarity'], errors='coerce').mean():.3f}")
c5.metric("Avg subjectivity", f"{pd.to_numeric(fdf['subjectivity'], errors='coerce').mean():.3f}")

st.divider()

# Trend (share over time)
st.subheader("Trend over time")

if trend_metric == "Share (stacked)":
    trend = sentiment_share_over_time(fdf, freq=freq_code)
    fig_trend = px.area(trend, x="date", y="share", color="class")
    fig_trend.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_trend, width='stretch')

elif trend_metric == "Count (stacked)":
    trend = sentiment_count_over_time(fdf, freq=freq_code)
    fig_trend = px.area(trend, x="date", y="count", color="class")
    st.plotly_chart(fig_trend, width='stretch')

else:  # Polarity mean (line)
    trend = polarity_mean_over_time(fdf, freq=freq_code)
    fig_trend = px.line(trend, x="date", y="polarity_mean")
    st.plotly_chart(fig_trend, width='stretch')

st.divider()

# Breakdown charts
left, right = st.columns(2)

with left:
    st.subheader("Sentiment distribution by source (Top 15 by volume)")
    top_sources = fdf["source"].value_counts().head(15).index
    tmp = fdf[fdf["source"].isin(top_sources)]
    fig_source = px.histogram(tmp, x="source", color="class", barmode="group")
    st.plotly_chart(fig_source, width='stretch')

with right:
    st.subheader("Sentiment distribution by subject (Top 15 by volume)")
    top_subjects = fdf["subject"].value_counts().head(15).index
    tmp2 = fdf[fdf["subject"].isin(top_subjects)]
    fig_subject = px.histogram(tmp2, x="subject", color="class", barmode="group")
    st.plotly_chart(fig_subject, width='stretch')





st.divider()
st.subheader("Drill-down")

drill_mode = st.radio("Drill by", options=["source", "subject"], horizontal=True)
if drill_mode == "source":
    options = top_source_tbl["source"].tolist() if len(top_source_tbl) else []
else:
    options = top_subject_tbl["subject"].tolist() if len(top_subject_tbl) else []

selected = st.selectbox(f"Select a {drill_mode}", options=options)

if selected:
    ddf = fdf[fdf[drill_mode] == selected].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(ddf):,}")
    c2.metric("Negative share", f"{(ddf['class'].eq('negative').mean() if len(ddf) else 0):.1%}")
    c3.metric("Avg polarity", f"{pd.to_numeric(ddf['polarity'], errors='coerce').mean():.3f}")

    # distribution
    dist = ddf["class"].value_counts().reset_index()
    dist.columns = ["class", "count"]
    fig = px.bar(dist, x="class", y="count")
    st.plotly_chart(fig, width='stretch')

    # latest news samples
    st.markdown("### Latest samples")
    show_n = st.slider("Show latest N", 5, 50, 15)
    cols = [c for c in ["date", "source", "subject", "class", "polarity", "title", "url"] if c in ddf.columns]
    st.dataframe(ddf.sort_values("date", ascending=False)[cols].head(show_n), width='stretch')
