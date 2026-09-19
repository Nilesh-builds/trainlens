import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent.parent

st.set_page_config(
    page_title="TrainLens — AI Training Data Quality",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Professional CSS ──────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg:       #0f0f23;    --bg2:      #1a1a3e;
    --surface:  #1e1e44;    --surface2: #2a2a5a;
    --border:   #333366;    --text:     #e0e0ff;
    --text2:    #8888aa;    --accent:   #00d4ff;
    --accent2:  #7b68ee;    --green:    #00e676;
    --red:      #ff5252;    --orange:   #ffab40;
    --yellow:   #ffd740;
}

[data-testid="stAppViewContainer"], .main { background: var(--bg) !important; }
[data-testid="stHeader"] { background: var(--bg) !important; }
[data-testid="stSidebar"] { background: var(--bg2) !important; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] .stMarkdown { color: var(--text); }
[data-testid="stSidebar"] label { color: var(--text) !important; }
[data-testid="stSidebar"] .stSelectbox, [data-testid="stSidebar"] .stMultiSelect { background: var(--surface); }

.metric-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
    padding: 1.3rem; text-align: center; transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover { transform: translateY(-2px); border-color: var(--accent); }
.metric-value { font-size: 2.2rem; font-weight: 700; color: var(--accent); }
.metric-label { font-size: 0.85rem; color: var(--text2); margin-top: 0.2rem; }

.gradient-text {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 2.6rem; font-weight: 700;
}

.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    color: var(--text2); padding: 0.6rem 1.5rem; font-weight: 500; transition: all 0.2s;
}
.stTabs [data-baseweb="tab"]:hover { background: var(--surface2); color: var(--text); }
.stTabs [data-baseweb="tab-highlight"] { background: var(--accent) !important; color: #000 !important; border-radius: 8px; }
.stTabs [aria-selected="true"] { color: #000 !important; font-weight: 600; }

[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 8px; }
.stPlotlyChart { border-radius: 12px; overflow: hidden; }
h1, h2, h3 { color: var(--text) !important; font-family: 'Inter', sans-serif; }
div[data-testid="stVerticalBlock"] > div { gap: 0.8rem; }
</style>""", unsafe_allow_html=True)


# ── Cached Data Loading ───────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    raw_path = ROOT / "data" / "raw" / "customer_support_tickets.csv"
    if not raw_path.exists():
        return None, None, None, None, None, None
    raw = pd.read_csv(raw_path)
    labeled_path = ROOT / "data" / "labeled" / "labeled_tickets.csv"
    review_path = ROOT / "data" / "labeled" / "review_results.csv"
    labeled = pd.read_csv(labeled_path) if labeled_path.exists() else None
    review = pd.read_csv(review_path) if review_path.exists() else None

    # Split labeled into category and sentiment views
    cat = labeled.copy() if labeled is not None else None
    sent = labeled.copy() if labeled is not None else None

    analytics = None
    report_path = ROOT / "data" / "processed" / "evaluation_report.json"
    if report_path.exists():
        import json
        analytics = json.loads(report_path.read_text())
    return raw, cat, sent, review, analytics


def run_pipeline():
    import subprocess, sys
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_pipeline.py")],
        capture_output=True, text=True, cwd=str(ROOT),
    )


def ensure_demo_data():
    """Create demo artifacts on first launch in a clean deployment environment."""
    raw_path = ROOT / "data" / "raw" / "customer_support_tickets.csv"
    if raw_path.exists():
        return True

    env = os.environ.copy()
    env["USE_LLM"] = "false"
    with st.spinner("Preparing the demo dataset for this deployment..."):
        generated = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "generate_data.py")],
            capture_output=True, text=True, cwd=str(ROOT), env=env,
        )
        if generated.returncode != 0:
            st.error("Could not generate the demo dataset.")
            return False
        pipeline = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_pipeline.py")],
            capture_output=True, text=True, cwd=str(ROOT), env=env,
        )
    if pipeline.returncode != 0:
        st.error("Could not prepare demo analytics. Check the deployment logs.")
        return False
    return True


def load_validation_report():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from data_quality.validator import run_validation
    raw_path = ROOT / "data" / "raw" / "customer_support_tickets.csv"
    if raw_path.exists():
        return run_validation(str(raw_path))
    return None


def metric_card(value, label):
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div>'
        f'</div>', unsafe_allow_html=True,
    )


def quality_badge(score):
    if score >= 90:   color, emoji = "#00e676", "🟢"
    elif score >= 70: color, emoji = "#ffab40", "🟠"
    else:             color, emoji = "#ff5252", "🔴"
    return f'{emoji} <span style="color:{color};font-weight:700">{score:.1f}%</span>'


def plotly_config(fig):
    return dict(fig, config=dict(displayModeBar=True, displaylogo=False, modeBarButtonsToRemove=["lasso2d", "select2d"]))


# ── Session State (persistent across reruns) ──────────────────────
for key in ("pipeline_ran", "show_settings"):
    if key not in st.session_state:
        st.session_state[key] = False


# ══════════════════════════════════════════════════════════════════
#                           SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="gradient-text">🔷 TrainLens</div>', unsafe_allow_html=True)
    st.caption("AI Training Data Quality Platform")
    st.divider()
    st.subheader("⚙️ Pipeline Controls")

    if st.button("▶ Run Full Pipeline", width='stretch', type="primary"):
        with st.spinner("Running pipeline..."):
            result = run_pipeline()
        if result.returncode == 0:
            st.success("Pipeline complete!")
            st.session_state.pipeline_ran = True
            st.cache_data.clear()
            st.rerun()
        else:
            st.error(f"Pipeline failed (exit {result.returncode})")

    st.divider()
    st.subheader("📥 Export")
    st.download_button("📥 Download Raw Data (CSV)",
                       data=(ROOT / "data" / "raw" / "customer_support_tickets.csv").read_bytes()
                       if (ROOT / "data" / "raw" / "customer_support_tickets.csv").exists() else b"",
                       file_name="trainlens_raw_data.csv", mime="text/csv",
                       width='stretch')

    st.divider()
    st.subheader("ℹ️ About")
    st.markdown(
        "**TrainLens** is an AI-powered data quality and evaluation platform "
        "for customer support teams. It validates, labels, and evaluates training data "
        "to improve AI model performance."
    )
    st.markdown("**Version:** 1.0 | **Author:** Nilesh-builds")


# ══════════════════════════════════════════════════════════════════
#                           MAIN APP
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="gradient-text">🔷 TrainLens</div>', unsafe_allow_html=True)
st.caption("AI Training Data Quality & Evaluation Platform | Customer Support Analytics")
st.info("Demo mode: this deployment uses synthetic customer-support data. No private customer data is included.")

if not ensure_demo_data():
    st.stop()

raw, cat, sent, review, analytics = load_data()
quality_report = load_validation_report() if (ROOT / "data" / "raw" / "customer_support_tickets.csv").exists() else None

if raw is None:
    st.error("Data not found. Run the pipeline first.")
    st.info("Use the **Run Full Pipeline** button in the sidebar, or run: `python scripts/run_pipeline.py`")
    st.stop()

with st.sidebar:
    st.divider()
    st.subheader("🧾 Run Metadata")
    if cat is not None and len(cat) > 0:
        run_ids = cat["label_run_id"].dropna().unique().tolist() if "label_run_id" in cat else []
        models = cat["label_model"].dropna().unique().tolist() if "label_model" in cat else []
        st.caption(f"Run: {run_ids[0] if run_ids else 'legacy dataset'}")
        st.caption(f"Model: {', '.join(models) if models else 'unknown'}")
        if "labeled_at" in cat:
            st.caption(f"Labeled: {cat['labeled_at'].iloc[0]}")

# Filters apply to analytical views while the quality report stays run-level.
with st.sidebar:
    st.divider()
    st.subheader("🔎 Explore Filters")
    selected_channels = st.multiselect("Channels", sorted(raw["channel"].dropna().unique())) if "channel" in raw else []
    selected_categories = st.multiselect(
        "Predicted categories", sorted(cat["predicted_category"].dropna().unique()) if cat is not None else []
    )
    selected_sentiments = st.multiselect(
        "Predicted sentiments", sorted(cat["predicted_sentiment"].dropna().unique()) if cat is not None else []
    )
    selected_methods = st.multiselect(
        "Label methods", sorted(cat["label_method"].dropna().unique()) if cat is not None and "label_method" in cat else []
    )

if selected_channels:
    raw = raw[raw["channel"].isin(selected_channels)]
if cat is not None:
    cat_mask = pd.Series(True, index=cat.index)
    if selected_categories:
        cat_mask &= cat["predicted_category"].isin(selected_categories)
    if selected_sentiments:
        cat_mask &= cat["predicted_sentiment"].isin(selected_sentiments)
    if selected_methods and "label_method" in cat:
        cat_mask &= cat["label_method"].isin(selected_methods)
    cat = cat[cat_mask]
    sent = cat.copy()
    valid_ids = set(cat["conversation_id"])
    raw = raw[raw["conversation_id"].isin(valid_ids)]
    if review is not None:
        review = review[review["conversation_id"].isin(valid_ids)]


# ══════════════════════════════════════════════════════════════════
#                       TABS LAYOUT
# ══════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", "🔍 Data Quality", "📈 Analytics",
    "🏷️ AI Labeling", "📏 Evaluation", "👥 Review Queue"
])


# ──────────────────── TAB 1: OVERVIEW ─────────────────────────────
with tab1:
    st.subheader("📊 Platform Overview")

    # KPI cards
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1: metric_card(f"{len(raw):,}", "Total Tickets")
    with k2: metric_card(quality_badge(quality_report['overall_score']), "Quality Score") if quality_report else metric_card("—", "Quality")
    with k3: metric_card(f"{cat['predicted_category'].nunique() if cat is not None else 0}", "Categories")
    with k4: metric_card(f"{cat['prediction_confidence'].mean()*100:.1f}%", "Avg Confidence") if cat is not None else metric_card("—", "Confidence")
    with k5: metric_card(f"{review['status'].value_counts().get('approved', 0):,}", "Reviewed") if review is not None else metric_card("—", "Reviewed")
    with k6: metric_card(f"{len(raw) * 0.023:,.0f}", "Needs Review")

    if cat is not None and "label_method" in cat:
        st.subheader("Label Coverage")
        method_counts = cat["label_method"].value_counts()
        coverage_cols = st.columns(max(1, len(method_counts)))
        for column, (method, count) in zip(coverage_cols, method_counts.items()):
            with column:
                metric_card(f"{count / len(cat):.1%}", f"{method} labels ({count:,})")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Issue Categories")
        if cat is not None:
            cat_counts = cat['predicted_category'].value_counts().reset_index()
            cat_counts.columns = ["category", "count"]
            fig = px.bar(cat_counts, x="category", y="count",
                         color="category", template="plotly_dark",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
                              paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e0ff")
            st.plotly_chart(fig, width='stretch')
    with c2:
        st.subheader("Sentiment Distribution")
        if sent is not None:
            sent_counts = sent['predicted_sentiment'].value_counts().reset_index()
            sent_counts.columns = ["sentiment", "count"]
            fig = px.pie(sent_counts, names="sentiment", values="count",
                         template="plotly_dark",
                         color_discrete_sequence=px.colors.qualitative.Pastel,
                         hole=0.5)
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                              paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e0ff")
            fig.update_traces(textinfo="label+percent", textfont_size=13)
            st.plotly_chart(fig, width='stretch')

    # Channel performance
    st.subheader("Channel Performance")
    ch_col1, ch_col2 = st.columns(2)
    with ch_col1:
        if 'channel' in raw.columns:
            ch = raw.groupby('channel').agg(
                tickets=('conversation_id', 'count'),
                avg_length=('customer_message', lambda x: x.str.len().mean())
            ).reset_index().sort_values('tickets', ascending=False)
            fig = px.bar(ch, x="channel", y="tickets", color="tickets",
                         template="plotly_dark", color_continuous_scale="Viridis")
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                              paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e0ff", showlegend=False)
            st.plotly_chart(fig, width='stretch')
    with ch_col2:
        if 'agent_name' in raw.columns:
            agent_cs = raw.groupby('agent_name').agg(
                tickets=('conversation_id', 'count'),
                avg_msg_len=('customer_message', lambda x: x.str.len().mean())
            ).reset_index().sort_values('tickets', ascending=False).head(10)
            fig = px.bar(agent_cs, x="agent_name", y="tickets", color="tickets",
                         template="plotly_dark", color_continuous_scale="Teal")
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                              paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e0ff", showlegend=False)
            st.plotly_chart(fig, width='stretch')

    # Weekly trend
    st.subheader("📈 Weekly Trend")
    raw['created_at'] = pd.to_datetime(raw['created_at'], errors='coerce')
    weekly = raw.set_index('created_at').resample('W').agg(
        tickets=('conversation_id', 'count'),
        avg_msg=('customer_message', lambda x: x.str.len().mean())
    ).reset_index().dropna()
    if len(weekly) > 1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=weekly['created_at'], y=weekly['tickets'],
                                 mode='lines+markers', name='Tickets',
                                 line=dict(color='#00d4ff', width=2)))
        fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e0ff",
                          title="Ticket Volume Over Time", title_font_color="#e0e0ff",
                          xaxis_title="Week", yaxis_title="Tickets")
        st.plotly_chart(fig, width='stretch')


# ──────────────────── TAB 2: DATA QUALITY ─────────────────────────
with tab2:
    st.subheader("🔍 Data Quality Framework")

    if quality_report is not None:
        k1, k2, k3, k4 = st.columns(4)
        with k1: metric_card(quality_badge(quality_report['overall_score']), "Overall Score")
        with k2: metric_card(f"{quality_report['passed']}/{quality_report['total_checks']}", "Checks Passed")
        with k3: metric_card(f"{quality_report['failed']}", "Failed Checks")
        with k4: metric_card(f"{quality_report['critical_failures']}", "Critical Failures")

        st.divider()
        st.subheader("Dimension Scores")
        checks = quality_report['checks']
        dimensions = {}
        for check in checks:
            dim = check['dimension']
            if dim not in dimensions:
                dimensions[dim] = {"passed": 0, "total": 0, "score_sum": 0}
            dimensions[dim]["total"] += 1
            dimensions[dim]["score_sum"] += check['pass_rate']
            if check['passed']:
                dimensions[dim]["passed"] += 1

        dim_scores = []
        for dim, d in dimensions.items():
            score = d['score_sum'] / d['total']
            dim_scores.append({"dimension": dim, "score": score, "checks": d['total'], "passed": d['passed']})

        dim_df = pd.DataFrame(dim_scores).sort_values("score", ascending=False)
        fig = px.bar(dim_df, x="dimension", y="score", color="score",
                     template="plotly_dark",
                     color_continuous_scale=[[0, "#ff5252"], [0.5, "#ffab40"], [1, "#00e676"]],
                     range_color=[0, 100])
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#e0e0ff", showlegend=False,
                          yaxis_title="Score (%)", xaxis_title="Dimension")
        st.plotly_chart(fig, width='stretch')

        st.subheader("Check Details")
        check_rows = []
        for c in checks:
            check_rows.append({
                "Check": c['id'],
                "Dimension": c['dimension'],
                "Severity": c['severity'],
                "Status": "✅ PASS" if c['passed'] else "❌ FAIL",
                "Pass Rate": f"{c['pass_rate']:.1f}%",
                "Details": c['message'],
            })
        st.dataframe(pd.DataFrame(check_rows), width='stretch', hide_index=True)
    else:
        st.info("No quality report found. Run the pipeline first.")


# ──────────────────── TAB 3: ANALYTICS ────────────────────────────
with tab3:
    st.subheader("📈 Analytics Dashboard")

    if cat is not None:
        # Category and sentiment predictions are stored together in labeled_tickets.csv.
        merged = cat.copy()

        # Category heatmap
        st.subheader("Category × Sentiment Distribution")
        cat_sent = merged.groupby(['predicted_category', 'predicted_sentiment']).size().reset_index(name='count')
        pivot = cat_sent.pivot(index='predicted_category', columns='predicted_sentiment', values='count').fillna(0)
        fig = px.imshow(pivot, text_auto=True, template="plotly_dark",
                        color_continuous_scale="Viridis",
                        labels=dict(x="Sentiment", y="Category", color="Count"))
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#e0e0ff")
        st.plotly_chart(fig, width='stretch')

        # Top categories by confidence
        st.subheader("Average Confidence by Category")
        cat_conf = merged.groupby('predicted_category')['prediction_confidence'].mean().reset_index()
        cat_conf.columns = ["category", "avg_confidence"]
        cat_conf = cat_conf.sort_values("avg_confidence", ascending=True)
        fig = px.bar(cat_conf, x="avg_confidence", y="category", color="avg_confidence",
                     orientation="h", template="plotly_dark",
                     color_continuous_scale="Teal", range_color=[0, 1])
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#e0e0ff", showlegend=False)
        st.plotly_chart(fig, width='stretch')

        # Message length analysis
        st.subheader("Message Length Analysis")
        merged['msg_len'] = merged['customer_message'].str.len()
        ml_col1, ml_col2 = st.columns(2)
        with ml_col1:
            fig = px.box(merged, x='predicted_category', y='msg_len',
                         template="plotly_dark", color='predicted_category',
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              font_color="#e0e0ff", showlegend=False)
            st.plotly_chart(fig, width='stretch')
        with ml_col2:
            fig = px.histogram(merged, x='msg_len', nbins=30, template="plotly_dark",
                               color_discrete_sequence=["#00d4ff"])
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              font_color="#e0e0ff", xaxis_title="Message Length", yaxis_title="Count")
            st.plotly_chart(fig, width='stretch')
    else:
        st.info("Analytics data not available. Run the pipeline first.")


# ──────────────────── TAB 4: AI LABELING ──────────────────────────
with tab4:
    st.subheader("🏷️ AI Labeling Results")

    if cat is not None:
        # Confidence distribution
        st.subheader("Confidence Distribution")
        fig = px.histogram(cat, x='prediction_confidence', nbins=30,
                           template="plotly_dark", color_discrete_sequence=["#7b68ee"])
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#e0e0ff",
                          xaxis_title="Confidence", yaxis_title="Count",
                          bargap=0.05)
        st.plotly_chart(fig, width='stretch')

        # Confidence tiers
        cat['tier'] = pd.cut(cat['prediction_confidence'],
                             bins=[0, 0.5, 0.75, 0.9, 1.01],
                             labels=['Low (0-50%)', 'Medium (50-75%)', 'High (75-90%)', 'Very High (90%+)'])
        tier_counts = cat['tier'].value_counts().reset_index()
        tier_counts.columns = ["tier", "count"]
        tier_colors = {'Low (0-50%)': '#ff5252', 'Medium (50-75%)': '#ffab40',
                       'High (75-90%)': '#ffd740', 'Very High (90%+)': '#00e676'}
        fig = px.bar(tier_counts, x="tier", y="count", color="tier",
                     template="plotly_dark",
                     color_discrete_map=tier_colors)
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#e0e0ff", showlegend=False)
        st.plotly_chart(fig, width='stretch')

        # Low confidence samples
        st.subheader("⚠️ Low Confidence Samples (Review Candidates)")
        low_conf = cat[cat['prediction_confidence'] < 0.6].sort_values('prediction_confidence').head(20)
        if len(low_conf) > 0:
            display_cols = [c for c in ['conversation_id', 'customer_message', 'predicted_category',
                                        'prediction_confidence'] if c in low_conf.columns]
            st.dataframe(low_conf[display_cols].reset_index(drop=True), width='stretch', hide_index=True)
        else:
            st.success("No low-confidence samples found.")

        # Category distribution
        st.subheader("Label Distribution")
        cat_dist = cat['predicted_category'].value_counts().reset_index()
        cat_dist.columns = ["category", "count"]
        fig = px.pie(cat_dist, names="category", values="count",
                     template="plotly_dark", hole=0.4,
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#e0e0ff")
        fig.update_traces(textinfo="label+value", textfont_size=12)
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Labeling data not available. Run the pipeline first.")


# ──────────────────── TAB 5: EVALUATION ───────────────────────────
with tab5:
    st.subheader("📏 Model Evaluation")

    if cat is not None:
        # Overall metrics
        cat['correct'] = cat['predicted_category'] == cat['category']
        accuracy = cat['correct'].mean()
        categories = cat['predicted_category'].unique()

        k1, k2, k3, k4 = st.columns(4)
        with k1: metric_card(f"{accuracy:.1%}", "Category Accuracy")
        with k2: metric_card(f"{cat['prediction_confidence'].mean():.1%}", "Avg Confidence")
        with k3: metric_card(f"{len(categories)}", "Categories")
        with k4: metric_card(f"{len(cat):,}", "Predictions")

        st.divider()

        # Confidence calibration
        st.subheader("🎯 Confidence Calibration")
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
        cat['conf_bin'] = pd.cut(cat['prediction_confidence'], bins=bins)
        cal = cat.groupby('conf_bin', observed=True).agg(
            avg_conf=('prediction_confidence', 'mean'),
            accuracy=('correct', 'mean'),
            count=('conversation_id', 'count')
        ).dropna().reset_index()
        if len(cal) > 1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=cal['avg_conf'], y=cal['accuracy'], mode='lines+markers+text',
                text=cal['count'], textposition="top center", textfont_size=10,
                name='Accuracy vs Confidence', line=dict(color='#00d4ff', width=3),
                marker=dict(size=10, color='#7b68ee')
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode='lines', name='Perfect Calibration',
                line=dict(dash='dash', color='#ff5252', width=2)
            ))
            fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                              paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e0ff",
                              xaxis_title="Mean Predicted Confidence", yaxis_title="Actual Accuracy",
                              legend=dict(x=0.02, y=0.98))
            st.plotly_chart(fig, width='stretch')

        # Per-category accuracy
        st.subheader("Per-Category Accuracy")
        per_cat = cat.groupby('predicted_category').agg(
            accuracy=('correct', 'mean'),
            count=('conversation_id', 'count'),
            avg_conf=('prediction_confidence', 'mean')
        ).reset_index().sort_values("accuracy", ascending=True)
        fig = px.bar(per_cat, x="accuracy", y="predicted_category", orientation='h',
                     color="accuracy", template="plotly_dark",
                     color_continuous_scale=[[0, "#ff5252"], [0.5, "#ffab40"], [1, "#00e676"]],
                     range_color=[0, 1], text=per_cat['accuracy'].apply(lambda x: f"{x:.1%}"))
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#e0e0ff", showlegend=False, xaxis_title="Accuracy")
        st.plotly_chart(fig, width='stretch')

        # Confusion matrix
        st.subheader("📊 Confusion Matrix")
        cm = pd.crosstab(cat['category'], cat['predicted_category'], margins=True)
        fig = px.imshow(cm, text_auto=True, template="plotly_dark",
                        color_continuous_scale="Blues",
                        labels=dict(x="Predicted", y="True", color="Count"))
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#e0e0ff")
        st.plotly_chart(fig, width='stretch')

        # Confidence histogram
        st.subheader("Confidence Distribution by Correctness")
        fig = px.histogram(cat, x='prediction_confidence', color='correct',
                           barmode='overlay', nbins=30, template="plotly_dark",
                           color_discrete_map={True: '#00e676', False: '#ff5252'},
                           opacity=0.7)
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#e0e0ff", xaxis_title="Confidence", yaxis_title="Count")
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Evaluation data not available. Run the pipeline first.")


# ──────────────────── TAB 6: REVIEW QUEUE ─────────────────────────
with tab6:
    st.subheader("👥 Human-in-the-Loop Review Queue")

    if review is not None:
        pending_rows = review[review["status"].isin(["pending", "in_review"])]
        if len(pending_rows) > 0:
            st.subheader("✍️ Review One Prediction")
            review_row = pending_rows.sort_values("prediction_confidence").iloc[0]
            st.caption(f"Reviewing {review_row['conversation_id']} | confidence {review_row['prediction_confidence']:.1%}")
            st.info(review_row["customer_message"])
            with st.form("review_prediction_form"):
                form_category = st.selectbox("Human category", ["billing", "technical_support", "shipping", "product_inquiry", "cancellation", "refund"], index=0)
                form_sentiment = st.selectbox("Human sentiment", ["positive", "neutral", "negative"], index=1)
                form_reviewer = st.text_input("Reviewer", value="analyst")
                form_notes = st.text_area("Notes")
                submitted = st.form_submit_button("Save Review", type="primary")
            if submitted:
                review_path = ROOT / "data" / "labeled" / "review_results.csv"
                review_data = pd.read_csv(review_path)
                row_mask = review_data["conversation_id"] == review_row["conversation_id"]
                review_data.loc[row_mask, "human_category"] = form_category
                review_data.loc[row_mask, "human_sentiment"] = form_sentiment
                review_data.loc[row_mask, "reviewer"] = form_reviewer
                review_data.loc[row_mask, "notes"] = form_notes
                review_data.loc[row_mask, "status"] = "reviewed"
                review_data.loc[row_mask, "reviewed_at"] = datetime.now().isoformat()
                review_data.to_csv(review_path, index=False)
                st.cache_data.clear()
                st.success("Review saved.")
                st.rerun()

        # Stats
        reviewed = review[~review['status'].isin(['pending', 'in_review'])]
        pending = review[review['status'].isin(['pending', 'in_review'])]
        approved = review[review['status'].isin(['approved', 'reviewed'])]
        rejected = review[review['status'] == 'rejected']

        k1, k2, k3, k4, k5 = st.columns(5)
        with k1: metric_card(f"{len(review):,}", "Total Reviews")
        with k2: metric_card(f"{len(pending):,}", "Pending")
        with k3: metric_card(f"{len(approved):,}", "Approved")
        with k4: metric_card(f"{len(rejected):,}", "Rejected")
        with k5:
            agree = (reviewed['predicted_category'] == reviewed['human_category']).mean() * 100 if len(reviewed) > 0 else 0
            metric_card(f"{agree:.1f}%", "Human-AI Agreement")

        # Status donut
        st.subheader("Review Status")
        rev_col1, rev_col2 = st.columns(2)
        with rev_col1:
            status_counts = review['status'].value_counts().reset_index()
            status_counts.columns = ["status", "count"]
            status_colors = {"pending": "#ffab40", "approved": "#00e676", "rejected": "#ff5252", "in_review": "#00d4ff"}
            fig = px.pie(status_counts, names="status", values="count",
                         template="plotly_dark", hole=0.5,
                         color_discrete_map=status_colors)
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              font_color="#e0e0ff")
            fig.update_traces(textinfo="label+percent", textfont_size=13)
            st.plotly_chart(fig, width='stretch')

        # Agreement by category
        with rev_col2:
            if len(reviewed) > 0 and 'predicted_category' in reviewed.columns:
                reviewed = reviewed.copy()
                reviewed['category_agreement'] = (
                    reviewed['predicted_category'] == reviewed['human_category']
                )
                agree_by_cat = (
                    reviewed.groupby('predicted_category')['category_agreement']
                    .mean().mul(100).reset_index()
                )
                agree_by_cat.columns = ["category", "agreement"]
                fig = px.bar(agree_by_cat, x="category", y="agreement", color="agreement",
                             template="plotly_dark",
                             color_continuous_scale=[[0, "#ff5252"], [0.5, "#ffab40"], [1, "#00e676"]],
                             range_color=[0, 100])
                fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                  font_color="#e0e0ff", showlegend=False, yaxis_title="Agreement %")
                st.plotly_chart(fig, width='stretch')

        # Disagreement analysis
        if len(reviewed) > 0:
            st.subheader("🔍 Disagreement Analysis")
            disagree = reviewed[reviewed['predicted_category'] != reviewed['human_category']]
            if len(disagree) > 0:
                disagree_summary = disagree.groupby(['predicted_category', 'human_category']).size().reset_index(name='count')
                disagree_summary = disagree_summary.sort_values('count', ascending=False).head(10)
                fig = px.bar(disagree_summary, x='count', y='predicted_category',
                             color='human_category', orientation='h', template="plotly_dark",
                             color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                  font_color="#e0e0ff", title="Top Disagreements (Predicted → True)")
                st.plotly_chart(fig, width='stretch')
            else:
                st.success("No disagreements found — perfect agreement!")

        # Lowest confidence samples
        st.subheader("⚠️ Lowest Confidence Samples (Review Candidates)")
        low_conf = review.sort_values('prediction_confidence').head(20)
        display_cols = [c for c in ['conversation_id', 'customer_message', 'predicted_category',
                                    'human_category', 'prediction_confidence', 'status']
                        if c in low_conf.columns]
        st.dataframe(low_conf[display_cols].reset_index(drop=True), width='stretch', hide_index=True)

        # Review table
        with st.expander("📋 Full Review Results"):
            st.dataframe(review, width='stretch', hide_index=True)
    else:
        st.info("Review data not available. Run the pipeline first.")

