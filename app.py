import streamlit as st
import pandas as pd
import numpy as np

from config import CUSTOM_CSS
from data_processor import (
    load_data, load_from_sql, preprocess_data, get_encoded_data,
    get_churn_summary, get_categorical_churn_rates, SAMPLE_DATASETS,
)
import charts
from ml_models import (
    get_available_models, get_available_models_with_categories,
    get_category_for_model, train_and_evaluate, benchmark_models,
    MODEL_CATEGORIES, DEFAULT_TOP_5,
)
from llm_insights import (
    get_executive_summary, get_categorical_insights, get_numerical_insights,
    get_model_insights, get_comprehensive_recommendations,
    get_segment_deep_dive, _call_llm,
)

st.set_page_config(
    page_title="Customer Churn Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📊 Churn Analyzer")
    st.markdown("---")

    st.markdown("#### Data Source")
    data_source = st.radio(
        "Choose data source:",
        ["Sample Dataset", "Upload File", "SQL Database"],
        label_visibility="collapsed",
    )

    uploaded_file = None
    sample_name = ""
    sql_df = None

    if data_source == "Sample Dataset":
        sample_name = st.selectbox(
            "Select dataset:",
            list(SAMPLE_DATASETS.keys()),
            key="sample_select",
        )
    elif data_source == "Upload File":
        uploaded_file = st.file_uploader(
            "Upload your data file",
            type=["csv", "xlsx", "xls", "json", "parquet"],
        )
    else:
        st.caption("Connect to any SQL database (SQLite, PostgreSQL, MySQL, etc.)")
        conn_str = st.text_input(
            "Connection string",
            placeholder="sqlite:///data.db  or  postgresql://user:pass@host/db",
        )
        sql_query = st.text_input(
            "SQL query",
            placeholder="SELECT * FROM customers",
        )
        if conn_str and sql_query and st.button("🔌 Connect", type="primary"):
            sql_df = load_from_sql(conn_str, sql_query)
            if sql_df is not None:
                st.session_state["sql_data"] = sql_df
                st.success(f"Loaded {len(sql_df):,} rows")
        if "sql_data" in st.session_state:
            sql_df = st.session_state["sql_data"]

    st.markdown("---")
    st.markdown("#### 🤖 LLM Insights")
    llm_provider = st.radio("Provider:", ["OpenAI", "Gemini"], horizontal=True,
                            key="llm_provider")
    if llm_provider == "OpenAI":
        api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    else:
        api_key = st.text_input("Gemini API Key", type="password", placeholder="AIza...")
    provider = "gemini" if llm_provider == "Gemini" else "openai"
    if api_key:
        st.success(f"{llm_provider} key provided", icon="✅")
    else:
        st.info(f"Add a {llm_provider} API key for AI insights", icon="ℹ️")

    st.markdown("---")
    st.markdown(
        "<small>Built with Streamlit, Plotly & scikit-learn</small>",
        unsafe_allow_html=True,
    )

# ── Load & Preprocess Data ────────────────────────────────────────────────

if sql_df is not None:
    raw_df = sql_df
elif uploaded_file is not None:
    raw_df = load_data(uploaded_file=uploaded_file)
else:
    raw_df = load_data(sample_name=sample_name)

if raw_df is None:
    st.warning("Please upload a file, select a sample dataset, or connect to a database.")
    st.stop()

result = preprocess_data(raw_df)
if result is None or result[0] is None:
    st.stop()

df, meta = result
cat_cols = meta["categorical_cols"]
num_cols = meta["numeric_cols"]

df_encoded = get_encoded_data(df, cat_cols, num_cols)
summary = get_churn_summary(df, num_cols)

# ── Header ─────────────────────────────────────────────────────────────────

st.markdown("# 📊 Customer Churn Analysis")
st.markdown("*Automated exploratory analysis, predictive modeling, and AI-powered insights*")

if meta["id_cols_dropped"]:
    st.caption(f"Auto-detected & dropped ID columns: {', '.join(meta['id_cols_dropped'])}")
st.caption(f"Target: **Churn** | {len(cat_cols)} categorical features | "
           f"{len(num_cols)} numerical features | {len(df):,} rows")
st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "🏠 Dashboard",
    "🔍 Data Explorer",
    "📊 Categorical Analysis",
    "📈 Numerical Analysis",
    "🔗 Correlation",
    "🤖 Predictive Models",
    "💡 AI Insights",
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════

with tabs[0]:
    st.markdown("### Key Performance Indicators")

    kpi_cards = [
        ("Total Customers", f'{summary["total_customers"]:,}', ""),
        ("Churn Rate", f'{summary["churn_rate"]:.1f}%', "metric-red"),
    ]
    if "money_col" in summary:
        kpi_cards.append((
            f'At Risk ({summary["money_col"]})',
            f'{summary["revenue_at_risk"]:,.0f}',
            "metric-green",
        ))
    if "tenure_col" in summary:
        kpi_cards.append((
            f'Avg {summary["tenure_col"]} (Churned)',
            f'{summary["avg_tenure_churned"]:.1f}',
            "metric-orange",
        ))

    kpi_cols = st.columns(len(kpi_cards))
    for i, (label, value, css_class) in enumerate(kpi_cards):
        with kpi_cols[i]:
            st.markdown(
                f'<div class="metric-card {css_class}"><h3>{label}</h3>'
                f'<h1>{value}</h1></div>',
                unsafe_allow_html=True,
            )

    st.markdown("")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(charts.churn_donut(df), use_container_width=True, key="dash_donut")
    with col2:
        if cat_cols:
            st.plotly_chart(
                charts.churn_rate_by_category(df, cat_cols[0]),
                use_container_width=True, key="dash_first_cat",
            )

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Quick Stats: Churned vs Retained")
        stat_rows = [
            ("Count", f'{summary["churned"]:,}', f'{summary["retained"]:,}'),
        ]
        if "money_col" in summary:
            mc = summary["money_col"]
            stat_rows.append((
                f"Avg {mc}",
                f'{summary["avg_money_churned"]:.2f}',
                f'{summary["avg_money_retained"]:.2f}',
            ))
        if "tenure_col" in summary:
            tc = summary["tenure_col"]
            stat_rows.append((
                f"Avg {tc}",
                f'{summary["avg_tenure_churned"]:.1f}',
                f'{summary["avg_tenure_retained"]:.1f}',
            ))
        for col in num_cols[:5]:
            if col not in [summary.get("money_col"), summary.get("tenure_col")]:
                stat_rows.append((
                    f"Avg {col}",
                    f'{df[df["Churn"]=="Yes"][col].mean():.2f}',
                    f'{df[df["Churn"]=="No"][col].mean():.2f}',
                ))
        stats_df = pd.DataFrame(stat_rows, columns=["Metric", "Churned", "Retained"])
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    with col4:
        st.markdown("#### Top Churn Drivers (by churn rate)")
        driver_data = []
        for col in cat_cols:
            rates = get_categorical_churn_rates(df, col)
            if "Churn Rate (%)" in rates.columns and len(rates) > 0:
                max_row = rates.loc[rates["Churn Rate (%)"].idxmax()]
                driver_data.append({
                    "Feature": col,
                    "Category": str(max_row[col]),
                    "Churn Rate": f'{max_row["Churn Rate (%)"]:.1f}%',
                })
        if driver_data:
            driver_df = pd.DataFrame(driver_data)
            driver_df = driver_df.sort_values("Churn Rate", ascending=False).head(10)
            st.dataframe(driver_df, use_container_width=True, hide_index=True)

    if api_key:
        st.markdown("---")
        st.markdown("#### 🤖 AI Executive Summary")
        if "llm_exec" not in st.session_state:
            with st.spinner("Generating AI insights..."):
                import io
                buf = io.StringIO()
                df.info(buf=buf)
                st.session_state["llm_exec"] = get_executive_summary(
                    api_key, summary, buf.getvalue(), provider=provider
                )
        st.markdown(
            f'<div class="llm-box">{st.session_state["llm_exec"]}</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════
# TAB 2: DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════

with tabs[1]:
    st.markdown("### Data Explorer")

    sub1, sub2, sub3 = st.tabs(["📋 Raw Data", "📊 Statistics", "📐 Distributions"])

    with sub1:
        st.markdown(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
        st.dataframe(df.head(100), use_container_width=True, height=400)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Data Types**")
            st.dataframe(
                pd.DataFrame(df.dtypes, columns=["Type"]).reset_index()
                    .rename(columns={"index": "Column"}),
                use_container_width=True, hide_index=True,
            )
        with col_b:
            st.markdown("**Missing Values**")
            missing = df.isnull().sum()
            missing_df = pd.DataFrame({
                "Column": missing.index, "Missing": missing.values,
                "% Missing": (missing.values / len(df) * 100).round(2),
            })
            st.dataframe(
                missing_df[missing_df["Missing"] > 0] if missing.sum() > 0
                else missing_df.head(10),
                use_container_width=True, hide_index=True,
            )

    with sub2:
        st.markdown("**Numerical Summary**")
        st.dataframe(df[num_cols].describe().round(2) if num_cols
                     else pd.DataFrame(), use_container_width=True)

        st.markdown("**Categorical Summary**")
        cat_summary = []
        for col in cat_cols:
            if col in df.columns and len(df[col].dropna()) > 0:
                mode_val = df[col].mode().iloc[0]
                cat_summary.append({
                    "Column": col,
                    "Unique Values": df[col].nunique(),
                    "Most Common": mode_val,
                    "Most Common %": f"{(df[col] == mode_val).mean()*100:.1f}%",
                })
        if cat_summary:
            st.dataframe(pd.DataFrame(cat_summary), use_container_width=True,
                         hide_index=True)

    with sub3:
        st.markdown("**Select a feature to explore:**")
        all_features = cat_cols + num_cols
        sel_col = st.selectbox("Feature", all_features, key="explore_col")
        if sel_col in cat_cols:
            st.plotly_chart(charts.churn_by_category(df, sel_col),
                            use_container_width=True, key="expl_cat")
            st.plotly_chart(charts.churn_rate_by_category(df, sel_col),
                            use_container_width=True, key="expl_rate")
        else:
            st.plotly_chart(charts.numeric_distribution(df, sel_col),
                            use_container_width=True, key="expl_hist")
            st.plotly_chart(charts.numeric_box(df, sel_col),
                            use_container_width=True, key="expl_box")


# ══════════════════════════════════════════════════════════════════════════
# TAB 3: CATEGORICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.markdown("### Categorical Feature Analysis")
    st.markdown("How each categorical feature relates to churn.")

    if not cat_cols:
        st.info("No categorical features detected in this dataset.")
    else:
        display_cats = cat_cols[:12]
        st.plotly_chart(
            charts.categorical_grid(df, display_cats[:6],
                                    "Categorical Features vs Churn (Part 1)"),
            use_container_width=True, key="cat_grid_1",
        )

        if len(display_cats) > 6:
            st.plotly_chart(
                charts.categorical_grid(df, display_cats[6:12],
                                        "Categorical Features vs Churn (Part 2)"),
                use_container_width=True, key="cat_grid_2",
            )

        st.markdown("---")
        st.markdown("#### Churn Rate by Category")
        rate_cols = st.columns(min(len(cat_cols), 2))
        for i, col in enumerate(cat_cols[:6]):
            with rate_cols[i % 2]:
                st.plotly_chart(
                    charts.churn_rate_by_category(df, col),
                    use_container_width=True, key=f"cat_rate_{i}",
                )

        # Auto-generated insight
        top_drivers = []
        for col in cat_cols:
            rates = get_categorical_churn_rates(df, col)
            if "Churn Rate (%)" in rates.columns and len(rates) > 0:
                max_row = rates.loc[rates["Churn Rate (%)"].idxmax()]
                top_drivers.append((col, str(max_row[col]), max_row["Churn Rate (%)"]))
        top_drivers.sort(key=lambda x: x[2], reverse=True)

        if top_drivers:
            bullets = "".join(
                f"• <b>{feat}</b>: '{cat}' has {rate:.1f}% churn rate<br>"
                for feat, cat, rate in top_drivers[:5]
            )
            st.markdown(
                f'<div class="insight-box"><b>Top Churn Drivers:</b><br>{bullets}</div>',
                unsafe_allow_html=True,
            )

        if api_key:
            st.markdown("#### 🤖 AI Categorical Insights")
            if "llm_cat" not in st.session_state:
                with st.spinner("Analyzing categories..."):
                    cat_data = ""
                    for col in cat_cols[:10]:
                        rates = get_categorical_churn_rates(df, col)
                        cat_data += f"\n{col}:\n{rates.to_string(index=False)}\n"
                    st.session_state["llm_cat"] = get_categorical_insights(
                        api_key, cat_data, provider=provider
                    )
            st.markdown(
                f'<div class="llm-box">{st.session_state["llm_cat"]}</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════
# TAB 4: NUMERICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

with tabs[3]:
    st.markdown("### Numerical Feature Analysis")
    st.markdown("Distribution and relationship of numerical features with churn.")

    if not num_cols:
        st.info("No numerical features detected in this dataset.")
    else:
        for i in range(0, min(len(num_cols), 6), 2):
            cols_pair = num_cols[i:i + 2]
            row_cols = st.columns(len(cols_pair))
            for j, col in enumerate(cols_pair):
                with row_cols[j]:
                    st.plotly_chart(
                        charts.numeric_distribution(df, col),
                        use_container_width=True, key=f"num_dist_{i+j}",
                    )

        if len(num_cols) >= 2:
            st.markdown("---")
            st.markdown("#### Feature Relationships")
            sc1, sc2 = st.columns(2)
            with sc1:
                x_col = st.selectbox("X axis", num_cols, index=0, key="scatter_x")
            with sc2:
                y_col = st.selectbox("Y axis", num_cols,
                                     index=min(1, len(num_cols) - 1), key="scatter_y")
            st.plotly_chart(
                charts.numeric_scatter(df, x_col, y_col),
                use_container_width=True, key="num_scatter",
            )

        # Auto-generated insight
        insights = []
        for col in num_cols[:5]:
            churned_mean = df[df["Churn"] == "Yes"][col].mean()
            retained_mean = df[df["Churn"] == "No"][col].mean()
            diff_pct = ((churned_mean - retained_mean) / retained_mean * 100
                        if retained_mean != 0 else 0)
            direction = "higher" if diff_pct > 0 else "lower"
            insights.append(
                f"• <b>{col}</b>: Churned customers average "
                f"{abs(diff_pct):.1f}% {direction} ({churned_mean:.2f} vs {retained_mean:.2f})<br>"
            )
        if insights:
            st.markdown(
                f'<div class="insight-box"><b>Key Numerical Patterns:</b><br>'
                f'{"".join(insights)}</div>',
                unsafe_allow_html=True,
            )

        if api_key:
            st.markdown("#### 🤖 AI Numerical Insights")
            if "llm_num" not in st.session_state:
                with st.spinner("Analyzing numerical features..."):
                    num_data = ""
                    for col in num_cols[:8]:
                        num_data += (
                            f"\n{col}:\n"
                            f"  Churned: mean={df[df['Churn']=='Yes'][col].mean():.2f}, "
                            f"median={df[df['Churn']=='Yes'][col].median():.2f}\n"
                            f"  Retained: mean={df[df['Churn']=='No'][col].mean():.2f}, "
                            f"median={df[df['Churn']=='No'][col].median():.2f}\n"
                        )
                    st.session_state["llm_num"] = get_numerical_insights(
                        api_key, num_data, provider=provider
                    )
            st.markdown(
                f'<div class="llm-box">{st.session_state["llm_num"]}</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════
# TAB 5: CORRELATION
# ══════════════════════════════════════════════════════════════════════════

with tabs[4]:
    st.markdown("### Feature Correlation Analysis")
    st.markdown("Understanding relationships between features and churn.")

    corr_bar = charts.churn_correlation_bar(df_encoded)
    if corr_bar:
        st.plotly_chart(corr_bar, use_container_width=True, key="corr_bar")

    st.markdown("---")
    st.plotly_chart(charts.correlation_heatmap(df_encoded),
                    use_container_width=True, key="corr_heatmap")

    st.markdown("---")
    st.markdown(
        '<div class="insight-box">'
        '<b>Interpretation Guide:</b><br>'
        '• <b>Positive correlation</b> with Churn = feature increases with churn likelihood<br>'
        '• <b>Negative correlation</b> with Churn = feature is associated with retention<br>'
        '• Look for the strongest positive and negative correlations as key churn drivers'
        '</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════
# TAB 6: PREDICTIVE MODELS
# ══════════════════════════════════════════════════════════════════════════

with tabs[5]:
    st.markdown("### Predictive Modeling")
    st.markdown("Evaluate → Select → Predict: benchmark all models first, "
                "then train the best ones with the full binary-optimised pipeline.")

    phase1, phase2 = st.tabs(["📊 Phase 1: Evaluate & Select",
                               "🚀 Phase 2: Train & Predict"])

    with phase1:
        st.markdown("#### Step 1 — Benchmark All Models")
        st.markdown(
            '<div class="insight-box">'
            "Runs <b>5-fold stratified cross-validation</b> on every available model "
            "and ranks them by a composite score (ROC-AUC + F1 + PR-AUC) / 3. "
            "The top 5 are auto-selected for the prediction phase."
            "</div>",
            unsafe_allow_html=True,
        )

        if st.button("🔬 Run Benchmark (all models)", type="primary",
                      use_container_width=True):
            with st.spinner("Benchmarking all models with 5-fold CV — "
                            "this takes ~60 seconds..."):
                bench_df = benchmark_models(df_encoded.to_json(), n_folds=5)
            st.session_state["benchmark"] = bench_df

        if "benchmark" in st.session_state:
            bench_df = st.session_state["benchmark"]
            st.markdown("---")
            st.markdown("#### Benchmark Results")

            top5 = bench_df.head(5)["Model"].tolist()
            st.success(f"**Recommended top 5:** {', '.join(top5)}")

            st.dataframe(
                bench_df.set_index("Model").style
                    .format("{:.4f}", subset=["ROC-AUC", "F1", "PR-AUC", "Composite"])
                    .format("{:.1f}", subset=["Time (s)"])
                    .highlight_max(axis=0,
                                   subset=["ROC-AUC", "F1", "PR-AUC", "Composite"],
                                   color="#d4edda"),
                use_container_width=True,
            )

            bc1, bc2 = st.columns(2)
            with bc1:
                st.plotly_chart(
                    charts.benchmark_ranking_chart(bench_df, top_n=5),
                    use_container_width=True, key="bench_rank",
                )
            with bc2:
                st.plotly_chart(
                    charts.benchmark_metrics_chart(bench_df),
                    use_container_width=True, key="bench_metrics",
                )

            st.markdown(
                '<div class="insight-box">'
                "<b>How to read this:</b> The composite score averages ROC-AUC, F1, "
                "and PR-AUC. Green bars are the recommended top 5. Proceed to "
                "<b>Phase 2</b> to train them with the full binary pipeline "
                "(threshold optimisation, class weighting, 10 metrics)."
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("Click the benchmark button above, or go directly to Phase 2 "
                    "which defaults to the 5 best models for this dataset.")

    with phase2:
        st.markdown("#### Step 2 — Select & Train Models")

        bench_top5 = (st.session_state["benchmark"].head(5)["Model"].tolist()
                      if "benchmark" in st.session_state else DEFAULT_TOP_5)

        models_by_cat = get_available_models_with_categories()
        cat_names = list(MODEL_CATEGORIES.keys())
        cat_columns = st.columns(len(cat_names))
        selected_models = []
        for idx, cat in enumerate(cat_names):
            with cat_columns[idx]:
                cat_models = [m for c, m in models_by_cat if c == cat]
                if cat_models:
                    st.markdown(f"**{cat}**")
                    for m in cat_models:
                        default_on = m in bench_top5
                        if st.checkbox(m, value=default_on, key=f"cb_{m}"):
                            selected_models.append(m)

        _mc1, _mc2 = st.columns([2, 1])
        with _mc1:
            n_cats = len({get_category_for_model(m) for m in selected_models})
            st.markdown(f"**{len(selected_models)}** model(s) selected across "
                        f"**{n_cats}** MECE categories")
        with _mc2:
            test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)

        churn_pct = summary["churn_rate"]
        st.markdown(
            '<div class="insight-box">'
            f"<b>Binary-optimised pipeline:</b> class-weight balancing for the "
            f"~{churn_pct:.1f}% minority class, Youden's J optimal threshold, "
            f"and 10 evaluation metrics including PR-AUC, MCC, F2 (recall-weighted), "
            f"balanced accuracy, and specificity."
            "</div>",
            unsafe_allow_html=True,
        )

        if selected_models and st.button("🚀 Train Selected Models", type="primary",
                                         use_container_width=True):
            with st.spinner("Training models with class-weight balancing..."):
                result = train_and_evaluate(
                    df_encoded.to_json(), selected_models, test_size,
                )

            if len(result) == 8:
                (results, roc_data, pr_data, confusion_matrices,
                 feature_importances, feature_names, reports,
                 threshold_info) = result
            else:
                st.error("Model training failed.")
                st.stop()

            if not results:
                st.error("No results returned.")
                st.stop()

            st.session_state["model_results"] = results
            st.session_state["roc_data"] = roc_data
            st.session_state["pr_data"] = pr_data
            st.session_state["confusion_matrices"] = confusion_matrices
            st.session_state["feature_importances"] = feature_importances
            st.session_state["feature_names"] = feature_names
            st.session_state["reports"] = reports
            st.session_state["threshold_info"] = threshold_info

        if "model_results" in st.session_state:
            results = st.session_state["model_results"]
            roc_data = st.session_state["roc_data"]
            pr_data = st.session_state["pr_data"]
            confusion_matrices = st.session_state["confusion_matrices"]
            feature_importances = st.session_state["feature_importances"]
            feature_names = st.session_state["feature_names"]
            reports = st.session_state["reports"]
            threshold_info = st.session_state["threshold_info"]

            st.markdown("---")
            st.markdown("#### Binary Classification Metrics (Optimal Threshold)")

            results_df = pd.DataFrame(results).T
            results_df.insert(0, "Category",
                              [get_category_for_model(m) for m in results_df.index])
            results_df.index.name = "Model"
            metric_cols = [c for c in results_df.columns if c != "Category"]

            best_auc = results_df["ROC-AUC"].idxmax()
            best_f1 = results_df["F1"].idxmax()
            best_recall = results_df["Recall (TPR)"].idxmax()

            bc1, bc2, bc3 = st.columns(3)
            bc1.metric("Best ROC-AUC", best_auc,
                       f"{results_df.loc[best_auc, 'ROC-AUC']:.4f}")
            bc2.metric("Best F1", best_f1,
                       f"{results_df.loc[best_f1, 'F1']:.4f}")
            bc3.metric("Best Recall", best_recall,
                       f"{results_df.loc[best_recall, 'Recall (TPR)']:.4f}")

            st.dataframe(
                results_df.style
                    .format("{:.4f}", subset=metric_cols)
                    .highlight_max(axis=0, subset=metric_cols, color="#d4edda")
                    .highlight_min(axis=0, subset=metric_cols, color="#f8d7da"),
                use_container_width=True,
            )
            st.plotly_chart(charts.model_comparison_chart(results),
                            use_container_width=True, key="ml_compare")

            st.markdown("---")
            st.markdown("#### Discrimination Curves")
            curve_c1, curve_c2 = st.columns(2)
            with curve_c1:
                roc_fixed = [(n, np.array(f), np.array(t), a)
                             for n, f, t, a in roc_data]
                st.plotly_chart(charts.roc_curves_chart(roc_fixed),
                                use_container_width=True, key="ml_roc")
            with curve_c2:
                pr_fixed = [(n, np.array(p), np.array(r), a)
                            for n, p, r, a in pr_data]
                st.plotly_chart(charts.pr_curves_chart(pr_fixed),
                                use_container_width=True, key="ml_pr")

            st.markdown(
                '<div class="insight-box">'
                f"<b>Why PR-AUC matters:</b> With ~{churn_pct:.1f}% churn "
                f"(imbalanced), ROC-AUC can be overly optimistic. PR-AUC focuses "
                f"on the positive (churn) class and better reflects how well the "
                f"model identifies actual churners."
                "</div>",
                unsafe_allow_html=True,
            )

            st.markdown("---")
            st.markdown("#### Threshold Optimisation")
            st.markdown("Default 0.5 is suboptimal for imbalanced binary targets. "
                        "Youden's J finds the threshold maximising TPR − FPR.")

            st.plotly_chart(
                charts.threshold_comparison_chart(threshold_info),
                use_container_width=True, key="ml_thresh_cmp",
            )

            sel_thresh = st.selectbox("Inspect threshold detail:",
                                      list(threshold_info.keys()),
                                      key="thresh_sel")
            if sel_thresh:
                ti = threshold_info[sel_thresh]
                tc1, tc2 = st.columns(2)
                with tc1:
                    st.plotly_chart(
                        charts.threshold_analysis_chart(
                            ti["y_true"], ti["y_prob"], sel_thresh),
                        use_container_width=True, key="ml_thresh_detail",
                    )
                with tc2:
                    st.plotly_chart(
                        charts.probability_distribution_chart(
                            ti["y_true"], ti["y_prob"], sel_thresh,
                            ti["optimal"]),
                        use_container_width=True, key="ml_prob_dist",
                    )

            st.markdown("---")
            st.markdown("#### Confusion Matrices (Optimal Threshold)")
            n_cm = len(confusion_matrices)
            cols_per_row = min(n_cm, 4)
            for row_start in range(0, n_cm, cols_per_row):
                cm_items = list(confusion_matrices.items())[
                    row_start:row_start + cols_per_row]
                cm_row = st.columns(len(cm_items))
                for i, (name, cm) in enumerate(cm_items):
                    with cm_row[i]:
                        st.plotly_chart(
                            charts.confusion_matrix_chart(np.array(cm), name),
                            use_container_width=True,
                            key=f"ml_cm_{row_start + i}",
                        )

            st.markdown("---")
            st.markdown("#### Classification Reports")
            for name, report in reports.items():
                with st.expander(f"📄 {name} ({get_category_for_model(name)})"):
                    report_df = pd.DataFrame(report).T
                    num_c = [c for c in report_df.columns
                             if report_df[c].dtype in ["float64", "float32"]]
                    st.dataframe(report_df.style.format("{:.3f}", subset=num_c),
                                 use_container_width=True)

            if feature_importances:
                st.markdown("---")
                st.markdown("#### Feature Importances")
                fi_tabs = st.tabs(list(feature_importances.keys()))
                for i, (name, imps) in enumerate(feature_importances.items()):
                    with fi_tabs[i]:
                        st.plotly_chart(
                            charts.feature_importance_chart(
                                imps, feature_names, name),
                            use_container_width=True, key=f"ml_fi_{i}",
                        )

            if api_key:
                st.markdown("---")
                st.markdown("#### 🤖 AI Model Interpretation")
                if "llm_model" not in st.session_state:
                    with st.spinner("Generating model insights..."):
                        st.session_state["llm_model"] = get_model_insights(
                            api_key, results, feature_importances,
                            feature_names, provider=provider,
                        )
                st.markdown(
                    f'<div class="llm-box">'
                    f'{st.session_state["llm_model"]}</div>',
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════
# TAB 7: AI INSIGHTS
# ══════════════════════════════════════════════════════════════════════════

with tabs[6]:
    st.markdown("### 💡 AI-Powered Deep Insights")

    if not api_key:
        st.warning(f"Please enter your {llm_provider} API key in the sidebar "
                   "to unlock AI-powered insights.")
        st.markdown("""
        **What you'll get with AI insights:**
        - Comprehensive churn recommendations with actionable strategies
        - Customer segment deep-dive analysis
        - Revenue impact assessment
        - Personalized retention playbook
        """)
    else:
        def _render_llm(key: str):
            if key in st.session_state:
                st.markdown(
                    f'<div class="llm-box">{st.session_state[key]}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("#### 📋 Comprehensive Retention Strategy")
        if st.button("Generate Retention Strategy", type="primary",
                      key="retention_btn"):
            with st.spinner("Building comprehensive strategy..."):
                top_factors = ""
                for col in cat_cols:
                    rates = get_categorical_churn_rates(df, col)
                    if "Churn Rate (%)" in rates.columns and len(rates) > 0:
                        max_row = rates.loc[rates["Churn Rate (%)"].idxmax()]
                        top_factors += (
                            f"- {col}: '{max_row[col]}' has "
                            f"{max_row['Churn Rate (%)']:.1f}% churn rate\n"
                        )
                st.session_state["llm_retention"] = \
                    get_comprehensive_recommendations(
                        api_key, summary, top_factors, provider=provider
                    )
        _render_llm("llm_retention")

        st.markdown("---")
        st.markdown("#### 🔬 Customer Segment Deep Dive")
        if st.button("Analyze Customer Segments", type="primary",
                      key="segment_btn"):
            with st.spinner("Performing segment analysis..."):
                segment_data = ""
                for col in cat_cols[:6]:
                    churn_rates = df.groupby(col)["Churn"].apply(
                        lambda x: f"{(x=='Yes').mean()*100:.1f}%"
                    ).to_string()
                    segment_data += f"\n{col}:\n{churn_rates}\n"

                for col in num_cols[:4]:
                    median_val = df[col].median()
                    high = df[df[col] > median_val]
                    low = df[df[col] <= median_val]
                    segment_data += (
                        f"\n{col} (above median {median_val:.1f}):\n"
                        f"  Churn rate: "
                        f"{(high['Churn']=='Yes').mean()*100:.1f}% "
                        f"(n={len(high)})\n"
                        f"{col} (at/below median):\n"
                        f"  Churn rate: "
                        f"{(low['Churn']=='Yes').mean()*100:.1f}% "
                        f"(n={len(low)})\n"
                    )
                st.session_state["llm_segments"] = get_segment_deep_dive(
                    api_key, segment_data, provider=provider
                )
        _render_llm("llm_segments")

        st.markdown("---")
        st.markdown("#### 🎯 Custom Question")
        custom_q = st.text_area(
            "Ask a specific question about the churn data:",
            placeholder="e.g., What strategies can reduce churn "
                        "among high-value customers?",
        )
        if custom_q and st.button("Ask AI", type="primary", key="custom_btn"):
            with st.spinner("Thinking..."):
                context = (
                    f"Dataset: Customer Churn "
                    f"({summary['total_customers']:,} customers)\n"
                    f"Churn Rate: {summary['churn_rate']:.1f}%\n"
                    f"Features: {', '.join(cat_cols[:8] + num_cols[:5])}\n"
                )
                if "money_col" in summary:
                    context += (
                        f"Avg {summary['money_col']}: churned "
                        f"{summary['avg_money_churned']:.2f} vs retained "
                        f"{summary['avg_money_retained']:.2f}\n"
                    )
                if "tenure_col" in summary:
                    context += (
                        f"Avg {summary['tenure_col']}: churned "
                        f"{summary['avg_tenure_churned']:.1f} vs retained "
                        f"{summary['avg_tenure_retained']:.1f}\n"
                    )
                prompt = f"Context:\n{context}\nQuestion: {custom_q}"
                st.session_state["llm_custom"] = _call_llm(
                    api_key, prompt, max_tokens=3000, provider=provider
                )
        _render_llm("llm_custom")
