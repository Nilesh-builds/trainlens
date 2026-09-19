"""TrainLens Dashboard - Streamlit application."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="TrainLens", page_icon=":bar_chart:", layout="wide")
st.title("TrainLens: AI Training Data Quality Platform")

DATA_RAW = ROOT / "data" / "raw"
DATA_LABELED = ROOT / "data" / "labeled"
DATA_PROCESSED = ROOT / "data" / "processed"


@st.cache_data
def load_data():
    raw = pd.read_csv(DATA_RAW / "customer_support_tickets.csv")
    labeled = pd.read_csv(DATA_LABELED / "labeled_tickets.csv")
    review = pd.read_csv(DATA_LABELED / "review_results.csv") if (DATA_LABELED / "review_results.csv").exists() else pd.DataFrame()
    eval_path = DATA_PROCESSED / "evaluation_report.json"
    eval_report = json.loads(eval_path.read_text()) if eval_path.exists() else {}
    return raw, labeled, review, eval_report


raw_df, labeled_df, review_df, eval_report = load_data()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview", "Data Quality", "Analytics", "AI Labeling", "Evaluation", "Review"
])

with tab1:
    st.header("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Conversations", f"{len(raw_df):,}")
    col2.metric("Categories", raw_df["category"].nunique())
    col3.metric("Avg Response Time", f"{raw_df['response_time_seconds'].mean():.0f}s")
    col4.metric("Resolution Rate", f"{raw_df['is_resolved'].mean()*100:.1f}%")

    st.subheader("Category Distribution")
    cat_dist = raw_df["category"].value_counts().reset_index()
    cat_dist.columns = ["category", "count"]
    fig = px.bar(cat_dist, x="category", y="count", color="category",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sentiment Distribution")
        sent_dist = raw_df["sentiment"].value_counts().reset_index()
        sent_dist.columns = ["sentiment", "count"]
        fig = px.pie(sent_dist, values="count", names="sentiment",
                     color_discrete_map={"positive": "#2ecc71", "neutral": "#3498db", "negative": "#e74c3c"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Channel Distribution")
        ch_dist = raw_df["channel"].value_counts().reset_index()
        ch_dist.columns = ["channel", "count"]
        fig = px.pie(ch_dist, values="count", names="channel",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Data Quality Report")
    validator_path = DATA_PROCESSED / "data_quality_report.json"
    if validator_path.exists():
        quality_report = json.loads(validator_path.read_text())
    else:
        from src.data_quality.validator import run_validation
        quality_report = run_validation(DATA_RAW / "customer_support_tickets.csv")

    score = quality_report["overall_score"]
    st.metric("Overall Quality Score", f"{score}%")
    st.progress(score / 100)

    st.subheader("Quality Checks")
    checks_df = pd.DataFrame(quality_report["checks"])
    checks_df["status"] = checks_df["passed"].apply(lambda x: "PASS" if x else "FAIL")
    st.dataframe(checks_df[["id", "dimension", "severity", "status", "pass_rate", "message"]],
                 use_container_width=True)

    st.subheader("Quality by Dimension")
    if not checks_df.empty:
        dim_scores = checks_df.groupby("dimension")["pass_rate"].mean().reset_index()
        fig = px.bar(dim_scores, x="dimension", y="pass_rate",
                     color="dimension", color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(yaxis_title="Pass Rate (%)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Analytics Dashboard")

    st.subheader("Sentiment by Category")
    sent_cat = labeled_df.groupby(["category", "sentiment"]).size().reset_index(name="count")
    fig = px.bar(sent_cat, x="category", y="count", color="sentiment",
                 barmode="group",
                 color_discrete_map={"positive": "#2ecc71", "neutral": "#3498db", "negative": "#e74c3c"})
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Response Time Distribution")
        fig = px.histogram(raw_df, x="response_time_seconds", nbins=30,
                          color_discrete_sequence=["#3498db"])
        fig.update_layout(xaxis_title="Response Time (seconds)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("CSAT Score Distribution")
        csat_dist = raw_df["customer_satisfaction"].value_counts().sort_index().reset_index()
        csat_dist.columns = ["score", "count"]
        fig = px.bar(csat_dist, x="score", y="count",
                     color_discrete_sequence=["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"])
        fig.update_layout(xaxis_title="CSAT Score", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Channel Performance")
    channel_perf = raw_df.groupby("channel").agg(
        avg_response=("response_time_seconds", "mean"),
        avg_csat=("customer_satisfaction", "mean"),
        resolution_rate=("is_resolved", "mean"),
        count=("conversation_id", "count")
    ).reset_index()
    st.dataframe(channel_perf.round(2), use_container_width=True)

with tab4:
    st.header("AI Labeling Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Labeled", f"{len(labeled_df):,}")
    col2.metric("Needs Review", f"{labeled_df['needs_review'].sum():,}")
    col3.metric("Review Rate", f"{labeled_df['needs_review'].mean()*100:.1f}%")

    st.subheader("Confidence Distribution")
    fig = px.histogram(labeled_df, x="prediction_confidence", nbins=20,
                      color="needs_review",
                      color_discrete_map={True: "#e74c3c", False: "#2ecc71"})
    fig.update_layout(xaxis_title="Confidence", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Predicted vs Actual Category")
    if "category" in labeled_df.columns:
        compare = labeled_df.groupby(["category", "predicted_category"]).size().reset_index(name="count")
        fig = px.sunburst(compare, path=["category", "predicted_category"], values="count",
                         color="count", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Uncertain Predictions")
    uncertain = labeled_df[labeled_df["needs_review"]].sort_values("prediction_confidence").head(20)
    st.dataframe(uncertain[["conversation_id", "customer_message", "predicted_category",
                            "predicted_sentiment", "prediction_confidence"]],
                 use_container_width=True)

with tab5:
    st.header("Evaluation Metrics")

    if eval_report:
        summary = eval_report.get("summary", {})
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Category Accuracy", f"{summary.get('category_accuracy', 0)}%")
        col2.metric("Category F1", f"{summary.get('category_f1', 0)}%")
        col3.metric("Sentiment Accuracy", f"{summary.get('sentiment_accuracy', 0)}%")
        col4.metric("Sentiment F1", f"{summary.get('sentiment_f1', 0)}%")

        st.subheader("Confusion Matrix - Category")
        cm_data = eval_report.get("category_confusion_matrix", {})
        if cm_data:
            labels = cm_data.get("labels", [])
            matrix = cm_data.get("matrix", [])
            fig = px.imshow(matrix, x=labels, y=labels, color_continuous_scale="Blues",
                           text_auto=True, aspect="auto")
            fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Confusion Matrix - Sentiment")
        cm_sent = eval_report.get("sentiment_confusion_matrix", {})
        if cm_sent:
            labels = cm_sent.get("labels", [])
            matrix = cm_sent.get("matrix", [])
            fig = px.imshow(matrix, x=labels, y=labels, color_continuous_scale="Oranges",
                           text_auto=True, aspect="auto")
            fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Confidence Analysis")
        conf_data = eval_report.get("confidence_analysis", [])
        if conf_data:
            conf_df = pd.DataFrame(conf_data)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=conf_df["confidence_bin"], y=conf_df["category_accuracy"],
                                name="Category Accuracy", marker_color="#3498db"))
            fig.add_trace(go.Bar(x=conf_df["confidence_bin"], y=conf_df["sentiment_accuracy"],
                                name="Sentiment Accuracy", marker_color="#e74c3c"))
            fig.update_layout(barmode="group", xaxis_title="Confidence Bin",
                            yaxis_title="Accuracy (%)")
            st.plotly_chart(fig, use_container_width=True)

with tab6:
    st.header("Human Review")

    if not review_df.empty:
        col1, col2, col3 = st.columns(3)
        reviewed = review_df[review_df["status"] == "reviewed"]
        col1.metric("Reviewed Items", f"{len(reviewed):,}")
        if not reviewed.empty:
            cat_agree = (reviewed["predicted_category"] == reviewed["human_category"]).mean() * 100
            sent_agree = (reviewed["predicted_sentiment"] == reviewed["human_sentiment"]).mean() * 100
            col2.metric("Category Agreement", f"{cat_agree:.1f}%")
            col3.metric("Sentiment Agreement", f"{sent_agree:.1f}%")

        st.subheader("Disagreements")
        disagreements = reviewed[
            (reviewed["predicted_category"] != reviewed["human_category"]) |
            (reviewed["predicted_sentiment"] != reviewed["human_sentiment"])
        ]
        if not disagreements.empty:
            st.dataframe(disagreements[["conversation_id", "customer_message",
                                       "predicted_category", "human_category",
                                       "predicted_sentiment", "human_sentiment"]],
                        use_container_width=True)
        else:
            st.info("No disagreements found")
    else:
        st.info("Run the pipeline first to generate review data")

st.sidebar.header("TrainLens")
st.sidebar.markdown("---")
st.sidebar.markdown("AI Training Data Quality Platform")
st.sidebar.markdown("---")
if st.sidebar.button("Run Pipeline"):
    with st.spinner("Running pipeline..."):
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "scripts" / "run_pipeline.py")], cwd=str(ROOT))
        st.rerun()
