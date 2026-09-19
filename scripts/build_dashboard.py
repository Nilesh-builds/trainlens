"""Build the dashboard file from parts."""
from pathlib import Path

ROOT = Path(__file__).parent.parent
dashboard_path = ROOT / "src" / "dashboard" / "app.py"

part3 = '''

# ==================== TAB 2: DATA QUALITY ====================
with tab2:
    st.subheader("Data Quality Report")
    from src.data_quality.validator import run_validation
    quality_report = run_validation(DATA_RAW / "customer_support_tickets.csv")
    score = quality_report["overall_score"]

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1: metric_card(f"{score}%", "Quality Score")
    with col2:
        color = "#34d399" if score > 90 else "#fbbf24" if score > 70 else "#f87171"
        st.markdown(f"""<div style="background:#1a1a3e;border-radius:12px;padding:1.5rem;text-align:center;">
            <div style="font-size:3rem;font-weight:800;background:linear-gradient(90deg,{color},#7b68ee);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{score}%</div>
            <div style="color:#8888aa;font-size:0.9rem;margin-top:0.5rem;">{quality_report['passed_checks']}/{quality_report['total_checks']} checks passed | {quality_report['critical_failures']} critical</div>
        </div>""", unsafe_allow_html=True)
    with col3: metric_card(f"{quality_report['failed_checks']}", "Failed Checks")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    checks_df = pd.DataFrame(quality_report["checks"])
    checks_df["status"] = checks_df["passed"].apply(lambda x: "PASS" if x else "FAIL")
    checks_df["color"] = checks_df["passed"].apply(lambda x: "#34d399" if x else "#f87171")

    fig = go.Figure()
    for _, row in checks_df.iterrows():
        fig.add_trace(go.Bar(x=[row["pass_rate"]], y=[row["id"]], orientation="h",
            marker=dict(color=row["color"]), text=[f'{row["pass_rate"]}%'], textposition="inside", showlegend=False))
    fig.update_layout(**DARK_LAYOUT, height=400, xaxis=dict(title="Pass Rate (%)", range=[0, 105], **GRID_STYLE),
        yaxis=dict(showgrid=False, autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(checks_df[["id", "dimension", "severity", "status", "pass_rate", "message"]], use_container_width=True, height=300)

    st.subheader("Quality by Dimension")
    dim_scores = checks_df.groupby("dimension")["pass_rate"].mean().reset_index()
    fig = go.Figure(go.Bar(x=dim_scores["dimension"], y=dim_scores["pass_rate"],
        marker=dict(color=CHART_COLORS["gradient"][:len(dim_scores)], line=dict(color="white", width=0.5)),
        text=dim_scores["pass_rate"].round(1).astype(str) + "%", textposition="outside"))
    fig.update_layout(**DARK_LAYOUT, height=350, xaxis=dict(showgrid=False), yaxis=dict(title="Pass Rate (%)", range=[0, 110], **GRID_STYLE))
    st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 3: ANALYTICS ====================
with tab3:
    st.subheader("Analytics Dashboard")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Sentiment by Category**")
        sent_cat = filtered_df.groupby(["category", "sentiment"]).size().reset_index(name="count")
        fig = px.bar(sent_cat, x="category", y="count", color="sentiment", barmode="stack",
            color_discrete_map=CHART_COLORS["sentiment"])
        fig.update_layout(**DARK_LAYOUT, height=380, xaxis=dict(showgrid=False), yaxis=dict(**GRID_STYLE))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Response Time Distribution**")
        fig = go.Figure(go.Histogram(x=filtered_df["response_time_seconds"], nbinsx=30,
            marker=dict(color="#00d4ff", line=dict(color="white", width=0.5))))
        fig.update_layout(**DARK_LAYOUT, height=380, xaxis=dict(title="Response Time (s)", showgrid=False), yaxis=dict(**GRID_STYLE))
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**CSAT Score Breakdown**")
        csat_dist = filtered_df["customer_satisfaction"].value_counts().sort_index().reset_index()
        csat_dist.columns = ["score", "count"]
        fig = go.Figure(go.Bar(x=csat_dist["score"], y=csat_dist["count"],
            marker=dict(color=["#f87171", "#fb923c", "#fbbf24", "#34d399", "#22c55e"]),
            text=csat_dist["count"], textposition="outside"))
        fig.update_layout(**DARK_LAYOUT, height=350, xaxis=dict(title="CSAT Score", showgrid=False), yaxis=dict(**GRID_STYLE))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Hourly Ticket Volume**")
        if "created_at" in filtered_df.columns:
            filtered_df["hour"] = pd.to_datetime(filtered_df["created_at"], errors="coerce").dt.hour
            hourly = filtered_df.dropna(subset=["hour"]).groupby("hour").size().reset_index(name="count")
            fig = go.Figure(go.Scatter(x=hourly["hour"], y=hourly["count"], mode="lines+markers",
                line=dict(color="#00d4ff", width=3, shape="spline"), marker=dict(size=8, color="#00d4ff"),
                fill="tozeroy", fillcolor="rgba(0,212,255,0.1)"))
            fig.update_layout(**DARK_LAYOUT, height=350, xaxis=dict(title="Hour of Day", showgrid=False), yaxis=dict(**GRID_STYLE))
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Agents by Volume")
    agent_perf = filtered_df.groupby("agent_id").agg(
        tickets=("conversation_id", "count"), avg_csat=("customer_satisfaction", "mean"),
        avg_response=("response_time_seconds", "mean"), resolution=("is_resolved", "mean")
    ).reset_index().sort_values("tickets", ascending=False).head(15)
    agent_perf["resolution"] = (agent_perf["resolution"] * 100).round(1)

    fig = go.Figure(go.Bar(x=agent_perf["agent_id"], y=agent_perf["tickets"],
        marker=dict(color=agent_perf["avg_csat"], colorscale="Viridis", colorbar=dict(title="CSAT")),
        text=agent_perf["tickets"], textposition="outside"))
    fig.update_layout(**DARK_LAYOUT, height=350, xaxis=dict(showgrid=False), yaxis=dict(**GRID_STYLE))
    st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 4: AI LABELING ====================
with tab4:
    st.subheader("AI Labeling Results")

    col1, col2, col3, col4 = st.columns(4)
    with col1: metric_card(f"{len(labeled_df):,}", "Total Labeled")
    with col2: metric_card(f"{labeled_df['needs_review'].sum():,}", "Needs Review")
    with col3: metric_card(f"{labeled_df['needs_review'].mean()*100:.1f}%", "Review Rate")
    with col4: metric_card(f"{labeled_df['prediction_confidence'].mean():.3f}", "Avg Confidence")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Confidence Distribution**")
        fig = go.Figure()
        for needs_rev, color, label in [(True, "#f87171", "Needs Review"), (False, "#34d399", "Confident")]:
            subset = labeled_df[labeled_df["needs_review"] == needs_rev]
            fig.add_trace(go.Histogram(x=subset["prediction_confidence"], nbinsx=20, name=label,
                marker=dict(color=color, line=dict(color="white", width=0.5)), opacity=0.8))
        fig.update_layout(**DARK_LAYOUT, height=380, barmode="stack",
            xaxis=dict(title="Confidence", showgrid=False), yaxis=dict(**GRID_STYLE),
            legend=dict(x=0.7, y=0.95, bgcolor="rgba(0,0,0,0.5)"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Predicted Category Distribution**")
        pred_cat = labeled_df["predicted_category"].value_counts().reset_index()
        pred_cat.columns = ["category", "count"]
        fig = go.Figure(go.Bar(x=pred_cat["category"], y=pred_cat["count"],
            marker=dict(color=CHART_COLORS["gradient"][:len(pred_cat)]),
            text=pred_cat["count"], textposition="outside"))
        fig.update_layout(**DARK_LAYOUT, height=380, xaxis=dict(showgrid=False), yaxis=dict(**GRID_STYLE))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Lowest Confidence Predictions (Review Candidates)")
    uncertain = labeled_df[labeled_df["needs_review"]].sort_values("prediction_confidence").head(25)
    fig = go.Figure(go.Bar(x=uncertain["conversation_id"], y=uncertain["prediction_confidence"],
        marker=dict(color=uncertain["prediction_confidence"], colorscale="Reds"),
        text=uncertain["prediction_confidence"].round(3), textposition="outside"))
    fig.update_layout(**DARK_LAYOUT, height=300, xaxis=dict(showgrid=False),
        yaxis=dict(title="Confidence", **GRID_STYLE))
    st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 5: EVALUATION ====================
with tab5:
    st.subheader("Evaluation Metrics")

    if eval_report:
        summary = eval_report.get("summary", {})
        col1, col2, col3, col4 = st.columns(4)
        with col1: metric_card(f"{summary.get('category_accuracy', 0)}%", "Category Accuracy")
        with col2: metric_card(f"{summary.get('category_f1', 0)}%", "Category F1")
        with col3: metric_card(f"{summary.get('sentiment_accuracy', 0)}%", "Sentiment Accuracy")
        with col4: metric_card(f"{summary.get('sentiment_f1', 0)}%", "Sentiment F1")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Confusion Matrix - Category**")
            cm_data = eval_report.get("category_confusion_matrix", {})
            if cm_data:
                labels = cm_data.get("labels", [])
                matrix = cm_data.get("matrix", [])
                fig = px.imshow(matrix, x=labels, y=labels, color_continuous_scale="Blues", text_auto=True, aspect="auto")
                fig.update_layout(**DARK_LAYOUT, height=400, xaxis=dict(title="Predicted"), yaxis=dict(title="Actual"))
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Confusion Matrix - Sentiment**")
            cm_sent = eval_report.get("sentiment_confusion_matrix", {})
            if cm_sent:
                labels = cm_sent.get("labels", [])
                matrix = cm_sent.get("matrix", [])
                fig = px.imshow(matrix, x=labels, y=labels, color_continuous_scale="Oranges", text_auto=True, aspect="auto")
                fig.update_layout(**DARK_LAYOUT, height=400, xaxis=dict(title="Predicted"), yaxis=dict(title="Actual"))
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Confidence Calibration")
        conf_data = eval_report.get("confidence_analysis", [])
        if conf_data:
            conf_df = pd.DataFrame(conf_data)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=conf_df["confidence_bin"], y=conf_df["category_accuracy"],
                mode="lines+markers", name="Category", line=dict(color="#00d4ff", width=3), marker=dict(size=10)))
            fig.add_trace(go.Scatter(x=conf_df["confidence_bin"], y=conf_df["sentiment_accuracy"],
                mode="lines+markers", name="Sentiment", line=dict(color="#ff6b9d", width=3), marker=dict(size=10)))
            fig.update_layout(**DARK_LAYOUT, height=380, xaxis=dict(title="Confidence Bin", showgrid=False),
                yaxis=dict(title="Accuracy (%)", **GRID_STYLE), legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0.5)"))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Top Errors")
        errors = eval_report.get("top_errors", [])
        if errors:
            st.dataframe(pd.DataFrame(errors), use_container_width=True, height=350)
    else:
        st.info("Run the pipeline first to generate evaluation data")

# ==================== TAB 6: REVIEW ====================
with tab6:
    st.subheader("Human Review")

    if not review_df.empty:
        reviewed = review_df[review_df["status"] == "reviewed"]
        col1, col2, col3, col4 = st.columns(4)
        with col1: metric_card(f"{len(reviewed):,}", "Reviewed Items")
        if not reviewed.empty:
            cat_agree = (reviewed["predicted_category"] == reviewed["human_category"]).mean() * 100
            sent_agree = (reviewed["predicted_sentiment"] == reviewed["human_sentiment"]).mean() * 100
            with col2: metric_card(f"{cat_agree:.1f}%", "Category Agreement")
            with col3: metric_card(f"{sent_agree:.1f}%", "Sentiment Agreement")
            with col4: metric_card(f"{len(review_df) - len(reviewed):,}", "Pending Review")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Review Status**")
            status_dist = review_df["status"].value_counts().reset_index()
            status_dist.columns = ["status", "count"]
            fig = go.Figure(go.Pie(labels=status_dist["status"], values=status_dist["count"], hole=0.6,
                marker=dict(colors=["#34d399", "#fbbf24"][:len(status_dist)]),
                textinfo="label+percent", textfont=dict(size=12)))
            fig.update_layout(**DARK_LAYOUT, height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Agreement by Category**")
            if not reviewed.empty:
                agree_by_cat = reviewed.groupby("category").apply(
                    lambda x: (x["predicted_category"] == x["human_category"]).mean() * 100).reset_index()
                agree_by_cat.columns = ["category", "agreement"]
                fig = go.Figure(go.Bar(x=agree_by_cat["category"], y=agree_by_cat["agreement"],
                    marker=dict(color=CHART_COLORS["gradient"][:len(agree_by_cat)]),
                    text=agree_by_cat["agreement"].round(1).astype(str) + "%", textposition="outside"))
                fig.update_layout(**DARK_LAYOUT, height=350, xaxis=dict(showgrid=False),
                    yaxis=dict(title="Agreement (%)", range=[0, 110], **GRID_STYLE))
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Disagreements")
        disagreements = reviewed[
            (reviewed["predicted_category"] != reviewed["human_category"]) |
            (reviewed["predicted_sentiment"] != reviewed["human_sentiment"])
        ]
        if not disagreements.empty:
            st.dataframe(disagreements[["conversation_id", "customer_message",
                                       "predicted_category", "human_category",
                                       "predicted_sentiment", "human_sentiment"]],
                        use_container_width=True, height=400)
        else:
            st.info("No disagreements found")
    else:
        st.info("Run the pipeline first to generate review data")
'''

# Read existing part1+2
existing = dashboard_path.read_text(encoding="utf-8")

# Append part3
with open(dashboard_path, "a", encoding="utf-8") as f:
    f.write(part3)

print("Dashboard assembled successfully")
