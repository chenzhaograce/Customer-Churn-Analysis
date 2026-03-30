import streamlit as st
import pandas as pd
import numpy as np

from config import CUSTOM_CSS, CATEGORICAL_COLS, DEMOGRAPHIC_COLS, SERVICE_COLS, ACCOUNT_COLS
from data_processor import load_data, preprocess_data, get_encoded_data, get_churn_summary, get_categorical_churn_rates
import charts
from ml_models import (
    get_available_models, get_available_models_with_categories,
    get_category_for_model, train_and_evaluate, benchmark_models,
    MODEL_CATEGORIES, DEFAULT_TOP_5,
)
from llm_insights import (
    get_executive_summary, get_demographic_insights, get_service_insights,
    get_billing_insights, get_model_insights, get_comprehensive_recommendations,
    get_segment_deep_dive,
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
    data_source = st.radio("Choose data source:", ["Sample Dataset", "Upload CSV"], label_visibility="collapsed")

    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

    st.markdown("---")
    st.markdown("#### 🤖 LLM Insights")
    llm_provider = st.radio("Provider:", ["OpenAI", "Gemini"], horizontal=True, key="llm_provider")
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

# ── Load Data ──────────────────────────────────────────────────────────────

raw_df = load_data(uploaded_file)
if raw_df is None:
    st.warning("Please upload a CSV file to get started.")
    st.stop()

df = preprocess_data(raw_df)
df_encoded = get_encoded_data(df)
summary = get_churn_summary(df)

# ── Header ─────────────────────────────────────────────────────────────────

st.markdown("# 📊 Customer Churn Analysis")
st.markdown("*Automated exploratory analysis, predictive modeling, and AI-powered insights*")
st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "🏠 Dashboard",
    "🔍 Data Explorer",
    "👥 Demographics",
    "📡 Services",
    "💳 Billing & Contract",
    "📈 Correlation",
    "🤖 Predictive Models",
    "💡 AI Insights",
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════

with tabs[0]:
    st.markdown("### Key Performance Indicators")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="metric-card"><h3>Total Customers</h3>'
            f'<h1>{summary["total_customers"]:,}</h1></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card metric-red"><h3>Churn Rate</h3>'
            f'<h1>{summary["churn_rate"]:.1f}%</h1></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card metric-green"><h3>Revenue at Risk</h3>'
            f'<h1>${summary["revenue_at_risk"]:,.0f}/mo</h1></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="metric-card metric-orange"><h3>Avg Tenure (Churned)</h3>'
            f'<h1>{summary["avg_tenure_churned"]:.0f} mo</h1></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(charts.churn_donut(df), use_container_width=True, key="dash_donut")
    with col2:
        st.plotly_chart(charts.tenure_group_chart(df), use_container_width=True, key="dash_tenure_grp")

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### Quick Stats: Churned vs Retained")
        stats_df = pd.DataFrame({
            "Metric": [
                "Count", "Avg Monthly Charges", "Avg Total Charges",
                "Avg Tenure (months)", "% Senior Citizens",
            ],
            "Churned": [
                f'{summary["churned"]:,}',
                f'${summary["avg_monthly_churned"]:.2f}',
                f'${df[df["Churn"]=="Yes"]["TotalCharges"].mean():,.2f}',
                f'{summary["avg_tenure_churned"]:.1f}',
                f'{(df[df["Churn"]=="Yes"]["SeniorCitizen"]=="Yes").mean()*100:.1f}%',
            ],
            "Retained": [
                f'{summary["retained"]:,}',
                f'${summary["avg_monthly_retained"]:.2f}',
                f'${df[df["Churn"]=="No"]["TotalCharges"].mean():,.2f}',
                f'{summary["avg_tenure_retained"]:.1f}',
                f'{(df[df["Churn"]=="No"]["SeniorCitizen"]=="Yes").mean()*100:.1f}%',
            ],
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    with col4:
        st.markdown("#### Top Churn Drivers (by churn rate)")
        driver_data = []
        for col in CATEGORICAL_COLS:
            rates = get_categorical_churn_rates(df, col)
            if "Churn Rate (%)" in rates.columns:
                max_row = rates.loc[rates["Churn Rate (%)"].idxmax()]
                driver_data.append({
                    "Feature": col,
                    "Category": max_row[col],
                    "Churn Rate": f'{max_row["Churn Rate (%)"]:.1f}%',
                })
        driver_df = pd.DataFrame(driver_data)
        driver_df = driver_df.sort_values("Churn Rate", ascending=False).head(8)
        st.dataframe(driver_df, use_container_width=True, hide_index=True)

    if api_key:
        st.markdown("---")
        st.markdown("#### 🤖 AI Executive Summary")
        with st.spinner("Generating AI insights..."):
            import io
            buf = io.StringIO()
            df.info(buf=buf)
            insight = get_executive_summary(api_key, summary, buf.getvalue(), provider=provider)
        st.markdown(f'<div class="llm-box">{insight}</div>', unsafe_allow_html=True)


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
                pd.DataFrame(df.dtypes, columns=["Type"]).reset_index().rename(columns={"index": "Column"}),
                use_container_width=True, hide_index=True,
            )
        with col_b:
            st.markdown("**Missing Values**")
            missing = df.isnull().sum()
            missing_df = pd.DataFrame({"Column": missing.index, "Missing": missing.values,
                                       "% Missing": (missing.values / len(df) * 100).round(2)})
            st.dataframe(missing_df[missing_df["Missing"] > 0] if missing.sum() > 0
                         else missing_df.head(10), use_container_width=True, hide_index=True)

    with sub2:
        st.markdown("**Numerical Summary**")
        st.dataframe(df.describe().round(2), use_container_width=True)

        st.markdown("**Categorical Summary**")
        cat_summary = []
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                cat_summary.append({
                    "Column": col,
                    "Unique Values": df[col].nunique(),
                    "Most Common": df[col].mode().iloc[0],
                    "Most Common %": f"{(df[col] == df[col].mode().iloc[0]).mean()*100:.1f}%",
                })
        st.dataframe(pd.DataFrame(cat_summary), use_container_width=True, hide_index=True)

    with sub3:
        st.markdown("**Select a feature to explore:**")
        sel_col = st.selectbox("Feature", df.columns.tolist(), key="explore_col")
        if df[sel_col].dtype in ["object", "category"] or df[sel_col].nunique() < 10:
            st.plotly_chart(charts.churn_by_category(df, sel_col), use_container_width=True, key="expl_cat")
            st.plotly_chart(charts.churn_rate_by_category(df, sel_col), use_container_width=True, key="expl_rate")
        else:
            import plotly.express as px
            fig = px.histogram(df, x=sel_col, color="Churn", nbins=30, barmode="overlay",
                               opacity=0.7, marginal="box",
                               color_discrete_map={"Yes": "#E15759", "No": "#4E79A7"})
            st.plotly_chart(fig, use_container_width=True, key="expl_hist")


# ══════════════════════════════════════════════════════════════════════════
# TAB 3: DEMOGRAPHICS
# ══════════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.markdown("### Customer Demographics Analysis")
    st.markdown("Understanding how customer demographics relate to churn behavior.")

    st.plotly_chart(charts.demographic_grid(df), use_container_width=True, key="demo_grid")

    st.markdown("---")
    st.markdown("#### Churn Rates by Demographic Segment")
    d1, d2 = st.columns(2)
    with d1:
        st.plotly_chart(charts.churn_rate_by_category(df, "gender"), use_container_width=True, key="demo_gender")
        st.plotly_chart(charts.churn_rate_by_category(df, "Partner"), use_container_width=True, key="demo_partner")
    with d2:
        st.plotly_chart(charts.churn_rate_by_category(df, "SeniorCitizen"), use_container_width=True, key="demo_senior")
        st.plotly_chart(charts.churn_rate_by_category(df, "Dependents"), use_container_width=True, key="demo_dep")

    st.markdown("---")
    st.markdown('<div class="insight-box">'
                '<b>Key Observations:</b><br>'
                '• Gender has minimal impact on churn — rates are nearly equal for male and female customers.<br>'
                '• Senior citizens churn at significantly higher rates (~41%) compared to non-seniors (~24%).<br>'
                '• Customers without partners or dependents show higher churn tendencies, '
                'suggesting single customers are less "sticky".'
                '</div>', unsafe_allow_html=True)

    if api_key:
        st.markdown("#### 🤖 AI Demographic Insights")
        with st.spinner("Analyzing demographics..."):
            demo_data = ""
            for col in DEMOGRAPHIC_COLS:
                rates = get_categorical_churn_rates(df, col)
                demo_data += f"\n{col}:\n{rates.to_string(index=False)}\n"
            insight = get_demographic_insights(api_key, demo_data, provider=provider)
        st.markdown(f'<div class="llm-box">{insight}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 4: SERVICES
# ══════════════════════════════════════════════════════════════════════════

with tabs[3]:
    st.markdown("### Service Subscription Analysis")
    st.markdown("How service subscriptions correlate with customer retention.")

    st.plotly_chart(charts.internet_service_chart(df), use_container_width=True, key="svc_internet")
    st.plotly_chart(charts.phone_service_chart(df), use_container_width=True, key="svc_phone")

    st.markdown("---")
    st.markdown("#### Add-on Services vs Churn")
    st.plotly_chart(charts.services_churn_grid(df), use_container_width=True, key="svc_grid")

    st.markdown("---")
    st.plotly_chart(charts.num_services_chart(df), use_container_width=True, key="svc_num")

    st.markdown("---")
    st.markdown('<div class="insight-box">'
                '<b>Key Observations:</b><br>'
                '• Fiber optic internet users churn at much higher rates than DSL users — '
                'possibly due to pricing or service quality issues.<br>'
                '• Customers without protective services (Online Security, Tech Support, '
                'Device Protection) churn more frequently.<br>'
                '• More add-on services correlate with lower churn, suggesting bundling is effective.'
                '</div>', unsafe_allow_html=True)

    if api_key:
        st.markdown("#### 🤖 AI Service Insights")
        with st.spinner("Analyzing services..."):
            svc_data = ""
            for col in SERVICE_COLS:
                rates = get_categorical_churn_rates(df, col)
                svc_data += f"\n{col}:\n{rates.to_string(index=False)}\n"
            insight = get_service_insights(api_key, svc_data, provider=provider)
        st.markdown(f'<div class="llm-box">{insight}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 5: BILLING & CONTRACT
# ══════════════════════════════════════════════════════════════════════════

with tabs[4]:
    st.markdown("### Billing & Contract Analysis")

    b1, b2 = st.columns(2)
    with b1:
        st.plotly_chart(charts.contract_chart(df), use_container_width=True, key="bill_contract")
    with b2:
        st.plotly_chart(charts.billing_chart(df), use_container_width=True, key="bill_paperless")

    st.plotly_chart(charts.payment_method_chart(df), use_container_width=True, key="bill_payment")

    st.markdown("---")
    st.markdown("#### Charges Analysis")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(charts.monthly_charges_dist(df), use_container_width=True, key="bill_monthly")
    with c2:
        st.plotly_chart(charts.total_charges_dist(df), use_container_width=True, key="bill_total")

    st.plotly_chart(charts.tenure_dist(df), use_container_width=True, key="bill_tenure")
    st.plotly_chart(charts.charges_scatter(df), use_container_width=True, key="bill_scatter")

    st.markdown("---")
    st.markdown('<div class="insight-box">'
                '<b>Key Observations:</b><br>'
                '• Month-to-month contracts have drastically higher churn (~43%) vs one-year (~11%) and two-year (~3%).<br>'
                '• Electronic check payment users churn significantly more than other payment methods.<br>'
                '• Churned customers tend to have higher monthly charges but lower total charges (shorter tenure).<br>'
                '• Paperless billing customers churn more — possibly less engaged or price-sensitive.'
                '</div>', unsafe_allow_html=True)

    if api_key:
        st.markdown("#### 🤖 AI Billing Insights")
        with st.spinner("Analyzing billing patterns..."):
            bill_data = ""
            for col in ACCOUNT_COLS:
                rates = get_categorical_churn_rates(df, col)
                bill_data += f"\n{col}:\n{rates.to_string(index=False)}\n"
            bill_data += f"\nMonthly Charges (Churned): mean={df[df['Churn']=='Yes']['MonthlyCharges'].mean():.2f}, median={df[df['Churn']=='Yes']['MonthlyCharges'].median():.2f}"
            bill_data += f"\nMonthly Charges (Retained): mean={df[df['Churn']=='No']['MonthlyCharges'].mean():.2f}, median={df[df['Churn']=='No']['MonthlyCharges'].median():.2f}"
            bill_data += f"\nTenure (Churned): mean={df[df['Churn']=='Yes']['tenure'].mean():.1f} months"
            bill_data += f"\nTenure (Retained): mean={df[df['Churn']=='No']['tenure'].mean():.1f} months"
            insight = get_billing_insights(api_key, bill_data, provider=provider)
        st.markdown(f'<div class="llm-box">{insight}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 6: CORRELATION
# ══════════════════════════════════════════════════════════════════════════

with tabs[5]:
    st.markdown("### Feature Correlation Analysis")
    st.markdown("Understanding relationships between features and churn.")

    corr_bar = charts.churn_correlation_bar(df_encoded)
    if corr_bar:
        st.plotly_chart(corr_bar, use_container_width=True, key="corr_bar")

    st.markdown("---")
    st.plotly_chart(charts.correlation_heatmap(df_encoded), use_container_width=True, key="corr_heatmap")

    st.markdown("---")
    st.markdown('<div class="insight-box">'
                '<b>Interpretation Guide:</b><br>'
                '• <b>Positive correlation</b> with Churn = feature increases with churn likelihood<br>'
                '• <b>Negative correlation</b> with Churn = feature is associated with retention<br>'
                '• Contract length, tenure, and protective services typically show strong negative correlation with churn'
                '</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 7: PREDICTIVE MODELS
# ══════════════════════════════════════════════════════════════════════════

with tabs[6]:
    st.markdown("### Predictive Modeling")
    st.markdown("Evaluate → Select → Predict: benchmark all models first, "
                "then train the best ones with the full binary-optimised pipeline.")

    phase1, phase2 = st.tabs(["📊 Phase 1: Evaluate & Select", "🚀 Phase 2: Train & Predict"])

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1 — BENCHMARK
    # ══════════════════════════════════════════════════════════════════

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

        if st.button("🔬 Run Benchmark (all models)", type="primary", use_container_width=True):
            with st.spinner("Benchmarking all models with 5-fold CV — this takes ~60 seconds..."):
                bench_df = benchmark_models(df_encoded.to_json(), n_folds=5)
            st.session_state["benchmark"] = bench_df

        if "benchmark" in st.session_state:
            bench_df = st.session_state["benchmark"]

            st.markdown("---")
            st.markdown("#### Benchmark Results")

            top5 = bench_df.head(5)["Model"].tolist()
            st.success(f"**Recommended top 5:** {', '.join(top5)}")

            score_cols = ["ROC-AUC", "F1", "PR-AUC", "Composite", "Time (s)"]
            st.dataframe(
                bench_df.set_index("Model").style
                    .format("{:.4f}", subset=["ROC-AUC", "F1", "PR-AUC", "Composite"])
                    .format("{:.1f}", subset=["Time (s)"])
                    .highlight_max(axis=0, subset=["ROC-AUC", "F1", "PR-AUC", "Composite"],
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
                "<b>How to read this:</b> The composite score averages ROC-AUC, F1, and PR-AUC. "
                "Green bars are the recommended top 5. Proceed to <b>Phase 2</b> to train them "
                "with the full binary pipeline (threshold optimisation, class weighting, 10 metrics)."
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("Click the benchmark button above, or go directly to Phase 2 "
                    "which defaults to the 5 best models for this dataset.")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2 — TRAIN & PREDICT
    # ══════════════════════════════════════════════════════════════════

    with phase2:
        st.markdown("#### Step 2 — Select & Train Models")

        bench_top5 = (st.session_state["benchmark"].head(5)["Model"].tolist()
                      if "benchmark" in st.session_state else DEFAULT_TOP_5)

        models_by_cat = get_available_models_with_categories()
        cat_names = list(MODEL_CATEGORIES.keys())
        cat_cols = st.columns(len(cat_names))
        selected_models = []
        for idx, cat in enumerate(cat_names):
            with cat_cols[idx]:
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

        st.markdown(
            '<div class="insight-box">'
            "<b>Binary-optimised pipeline:</b> class-weight balancing for the ~26.5% minority class, "
            "Youden's J optimal threshold, and 10 evaluation metrics including PR-AUC, MCC, "
            "F2 (recall-weighted), balanced accuracy, and specificity."
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
                 feature_importances, feature_names, reports, threshold_info) = result
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

            # ── Metrics table ──────────────────────────────────────────
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

            # ── ROC + PR curves ────────────────────────────────────────
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
                "<b>Why PR-AUC matters:</b> With ~26.5% churn (imbalanced), ROC-AUC can be "
                "overly optimistic. PR-AUC focuses on the positive (churn) class and better "
                "reflects how well the model identifies actual churners."
                "</div>",
                unsafe_allow_html=True,
            )

            # ── Threshold analysis ─────────────────────────────────────
            st.markdown("---")
            st.markdown("#### Threshold Optimisation")
            st.markdown("Default 0.5 is suboptimal for imbalanced binary targets. "
                        "Youden's J finds the threshold maximising TPR − FPR.")

            st.plotly_chart(
                charts.threshold_comparison_chart(threshold_info),
                use_container_width=True, key="ml_thresh_cmp",
            )

            sel_thresh = st.selectbox("Inspect threshold detail:",
                                      list(threshold_info.keys()), key="thresh_sel")
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
                            ti["y_true"], ti["y_prob"], sel_thresh, ti["optimal"]),
                        use_container_width=True, key="ml_prob_dist",
                    )

            # ── Confusion matrices ─────────────────────────────────────
            st.markdown("---")
            st.markdown("#### Confusion Matrices (Optimal Threshold)")
            n_cm = len(confusion_matrices)
            cols_per_row = min(n_cm, 4)
            for row_start in range(0, n_cm, cols_per_row):
                cm_items = list(confusion_matrices.items())[row_start:row_start + cols_per_row]
                cm_row = st.columns(len(cm_items))
                for i, (name, cm) in enumerate(cm_items):
                    with cm_row[i]:
                        st.plotly_chart(
                            charts.confusion_matrix_chart(np.array(cm), name),
                            use_container_width=True, key=f"ml_cm_{row_start + i}",
                        )

            # ── Classification reports ─────────────────────────────────
            st.markdown("---")
            st.markdown("#### Classification Reports")
            for name, report in reports.items():
                with st.expander(f"📄 {name} ({get_category_for_model(name)})"):
                    report_df = pd.DataFrame(report).T
                    num_cols = [c for c in report_df.columns
                                if report_df[c].dtype in ["float64", "float32"]]
                    st.dataframe(report_df.style.format("{:.3f}", subset=num_cols),
                                 use_container_width=True)

            # ── Feature importances ────────────────────────────────────
            if feature_importances:
                st.markdown("---")
                st.markdown("#### Feature Importances")
                fi_tabs = st.tabs(list(feature_importances.keys()))
                for i, (name, imps) in enumerate(feature_importances.items()):
                    with fi_tabs[i]:
                        st.plotly_chart(
                            charts.feature_importance_chart(imps, feature_names, name),
                            use_container_width=True, key=f"ml_fi_{i}",
                        )

            # ── LLM interpretation ─────────────────────────────────────
            if api_key:
                st.markdown("---")
                st.markdown("#### 🤖 AI Model Interpretation")
                with st.spinner("Generating model insights..."):
                    insight = get_model_insights(
                        api_key, results, feature_importances, feature_names,
                        provider=provider)
                st.markdown(f'<div class="llm-box">{insight}</div>',
                            unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 8: AI INSIGHTS
# ══════════════════════════════════════════════════════════════════════════

with tabs[7]:
    st.markdown("### 💡 AI-Powered Deep Insights")

    if not api_key:
        st.warning(f"Please enter your {llm_provider} API key in the sidebar to unlock AI-powered insights.")
        st.markdown("""
        **What you'll get with AI insights:**
        - Comprehensive churn recommendations with actionable strategies
        - Customer segment deep-dive analysis
        - Revenue impact assessment
        - Personalized retention playbook
        """)
    else:
        def _render_llm(key: str):
            """Display a cached LLM response from session state."""
            if key in st.session_state:
                st.markdown(
                    f'<div class="llm-box">{st.session_state[key]}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("#### 📋 Comprehensive Retention Strategy")
        if st.button("Generate Retention Strategy", type="primary", key="retention_btn"):
            with st.spinner("Building comprehensive strategy..."):
                top_factors = ""
                for col in CATEGORICAL_COLS:
                    rates = get_categorical_churn_rates(df, col)
                    if "Churn Rate (%)" in rates.columns:
                        max_row = rates.loc[rates["Churn Rate (%)"].idxmax()]
                        top_factors += f"- {col}: '{max_row[col]}' has {max_row['Churn Rate (%)']:.1f}% churn rate\n"
                st.session_state["llm_retention"] = get_comprehensive_recommendations(
                    api_key, summary, top_factors, provider=provider
                )
        _render_llm("llm_retention")

        st.markdown("---")
        st.markdown("#### 🔬 Customer Segment Deep Dive")
        if st.button("Analyze Customer Segments", type="primary", key="segment_btn"):
            with st.spinner("Performing segment analysis..."):
                segment_data = f"""
Tenure Groups:
{df.groupby('tenure_group')['Churn'].apply(lambda x: f"{(x=='Yes').mean()*100:.1f}%").to_string()}

Contract x Internet:
{df.groupby(['Contract','InternetService'])['Churn'].apply(lambda x: f"{(x=='Yes').mean()*100:.1f}%").to_string()}

Senior Citizen x Contract:
{df.groupby(['SeniorCitizen','Contract'])['Churn'].apply(lambda x: f"{(x=='Yes').mean()*100:.1f}%").to_string()}

Number of Services Distribution:
{df.groupby('NumServices')['Churn'].apply(lambda x: f"{(x=='Yes').mean()*100:.1f}%").to_string()}

High-Value Customers (Monthly > $70):
Churn rate: {(df[df['MonthlyCharges']>70]['Churn']=='Yes').mean()*100:.1f}%
Count: {len(df[df['MonthlyCharges']>70])}

Low-Tenure Customers (< 12 months):
Churn rate: {(df[df['tenure']<12]['Churn']=='Yes').mean()*100:.1f}%
Count: {len(df[df['tenure']<12])}
"""
                st.session_state["llm_segments"] = get_segment_deep_dive(
                    api_key, segment_data, provider=provider
                )
        _render_llm("llm_segments")

        st.markdown("---")
        st.markdown("#### 🎯 Custom Question")
        custom_q = st.text_area("Ask a specific question about the churn data:",
                                placeholder="e.g., What strategies can reduce churn among fiber optic users?")
        if custom_q and st.button("Ask AI", type="primary", key="custom_btn"):
            with st.spinner("Thinking..."):
                from llm_insights import _call_llm  # noqa: F811
                context = f"""Dataset: Telco Customer Churn ({summary['total_customers']:,} customers)
Churn Rate: {summary['churn_rate']:.1f}%
Key facts: Fiber optic users churn most, month-to-month contracts have ~43% churn,
electronic check users churn most, senior citizens churn ~41%.
Avg monthly charges: churned ${summary['avg_monthly_churned']:.2f} vs retained ${summary['avg_monthly_retained']:.2f}.
Avg tenure: churned {summary['avg_tenure_churned']:.1f}mo vs retained {summary['avg_tenure_retained']:.1f}mo."""
                prompt = f"Context:\n{context}\n\nQuestion: {custom_q}"
                st.session_state["llm_custom"] = _call_llm(
                    api_key, prompt, max_tokens=3000, provider=provider
                )
        _render_llm("llm_custom")
