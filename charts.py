import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from config import CHURN_COLORS, CHURN_COLOR_SEQ, COLORS, VIVID_CATEGORICAL

LAYOUT_DEFAULTS = dict(
    template="plotly_white",
    font=dict(family="Inter, sans-serif", size=13),
    margin=dict(l=40, r=40, t=50, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=420,
)


def _apply_layout(fig, title=None, **kwargs):
    opts = {**LAYOUT_DEFAULTS, **kwargs}
    if title:
        opts["title"] = dict(text=title, x=0, xanchor="left", font_size=16)
    fig.update_layout(**opts)
    return fig


# ── Generic EDA charts ─────────────────────────────────────────────────────

def churn_donut(df):
    counts = df["Churn"].value_counts().reset_index()
    counts.columns = ["Churn", "Count"]
    fig = px.pie(
        counts, values="Count", names="Churn", hole=0.5,
        color="Churn", color_discrete_map=CHURN_COLORS,
    )
    fig.update_traces(textinfo="percent+label", textfont_size=14)
    return _apply_layout(fig, "Churn Distribution", height=380)


def churn_by_category(df, col, title=None):
    ct = df.groupby([col, "Churn"]).size().reset_index(name="Count")
    fig = px.bar(
        ct, x=col, y="Count", color="Churn", barmode="group",
        color_discrete_map=CHURN_COLORS, text_auto=True,
    )
    fig.update_traces(textposition="outside", textfont_size=11)
    return _apply_layout(fig, title or f"Churn by {col}")


def churn_rate_by_category(df, col, title=None):
    rates = df.groupby(col)["Churn"].apply(lambda x: (x == "Yes").mean() * 100).reset_index()
    rates.columns = [col, "Churn Rate (%)"]
    rates = rates.sort_values("Churn Rate (%)", ascending=True)
    fig = px.bar(
        rates, x="Churn Rate (%)", y=col, orientation="h",
        text=rates["Churn Rate (%)"].round(1),
        color="Churn Rate (%)", color_continuous_scale=["#4E79A7", "#E15759"],
    )
    fig.update_traces(textposition="outside", textfont_size=12)
    fig.update_coloraxes(showscale=False)
    return _apply_layout(fig, title or f"Churn Rate by {col}",
                         height=max(300, len(rates) * 45 + 100))


def categorical_grid(df, cols, title="Categorical Features vs Churn"):
    """Grid of bar charts showing churn by each categorical column."""
    n = len(cols)
    if n == 0:
        return go.Figure()
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=[c.replace("_", " ").title() for c in cols],
    )
    for i, col in enumerate(cols):
        r, c = divmod(i, ncols)
        ct = df.groupby([col, "Churn"]).size().reset_index(name="Count")
        for churn_val in ["No", "Yes"]:
            subset = ct[ct["Churn"] == churn_val]
            fig.add_trace(
                go.Bar(
                    x=subset[col].astype(str), y=subset["Count"], name=churn_val,
                    marker_color=CHURN_COLORS[churn_val],
                    showlegend=(i == 0), legendgroup=churn_val,
                    text=subset["Count"], textposition="outside",
                ),
                row=r + 1, col=c + 1,
            )
    h = max(400, nrows * 320)
    return _apply_layout(fig, title, height=h, barmode="group")


def numeric_distribution(df, col, title=None):
    """Histogram of a numeric column split by Churn."""
    fig = px.histogram(
        df, x=col, color="Churn", nbins=40,
        color_discrete_map=CHURN_COLORS, barmode="overlay", opacity=0.7,
        marginal="box",
    )
    return _apply_layout(fig, title or f"{col} Distribution by Churn", height=450)


def numeric_scatter(df, col_x, col_y, title=None):
    """Scatter of two numeric columns coloured by Churn."""
    fig = px.scatter(
        df, x=col_x, y=col_y, color="Churn",
        color_discrete_map=CHURN_COLORS, opacity=0.5,
    )
    return _apply_layout(fig, title or f"{col_x} vs {col_y}", height=450)


def numeric_box(df, col, title=None):
    """Box plot of a numeric column grouped by Churn."""
    fig = px.box(
        df, x="Churn", y=col, color="Churn",
        color_discrete_map=CHURN_COLORS,
    )
    return _apply_layout(fig, title or f"{col} by Churn Status", height=400)


# ── Correlation ────────────────────────────────────────────────────────────

def correlation_heatmap(df_encoded):
    numeric_df = df_encoded.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    churn_col = "Churn" if "Churn" in corr.columns else None
    if churn_col:
        churn_corr = corr[churn_col].drop(churn_col).sort_values(ascending=False)
        top_features = churn_corr.head(10).index.tolist() + churn_corr.tail(10).index.tolist()
        top_features = list(dict.fromkeys(top_features))
        cols_to_show = [churn_col] + top_features
        corr = corr.loc[cols_to_show, cols_to_show]

    fig = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
    )
    size = max(500, len(corr) * 30 + 100)
    return _apply_layout(fig, "Correlation Heatmap (Top Features)", height=size, width=size + 100)


def churn_correlation_bar(df_encoded):
    numeric_df = df_encoded.select_dtypes(include=[np.number])
    if "Churn" not in numeric_df.columns:
        return None
    corr = numeric_df.corr()["Churn"].drop("Churn").sort_values()
    colors = [COLORS["churned"] if v > 0 else COLORS["retained"] for v in corr.values]

    fig = go.Figure(go.Bar(
        x=corr.values, y=corr.index, orientation="h",
        marker_color=colors, text=corr.values.round(3), textposition="outside",
    ))
    return _apply_layout(
        fig, "Feature Correlation with Churn",
        height=max(400, len(corr) * 22 + 100), xaxis_title="Correlation Coefficient",
    )


# ── Model Benchmark ───────────────────────────────────────────────────────

def benchmark_ranking_chart(bench_df, top_n=5):
    df = bench_df.sort_values("Composite", ascending=True).copy()
    df["Selected"] = df["Rank"] <= top_n
    df["Color"] = df["Selected"].map({True: COLORS["accent"], False: "#cccccc"})

    fig = go.Figure(go.Bar(
        x=df["Composite"], y=df["Model"], orientation="h",
        marker_color=df["Color"],
        text=df["Composite"].round(4), textposition="outside",
    ))
    fig.update_xaxes(title="Composite Score  (ROC-AUC + F1 + PR-AUC) / 3")
    return _apply_layout(fig, f"Model Benchmark — Top {top_n} Highlighted",
                         height=max(350, len(df) * 38 + 80))


def benchmark_metrics_chart(bench_df):
    df = bench_df.sort_values("Rank")
    melted = df.melt(id_vars=["Model", "Rank"], value_vars=["ROC-AUC", "F1", "PR-AUC"],
                     var_name="Metric", value_name="Score")
    fig = px.bar(
        melted, x="Model", y="Score", color="Metric", barmode="group",
        text=melted["Score"].round(3),
        color_discrete_sequence=[COLORS["retained"], COLORS["accent"], COLORS["churned"]],
    )
    fig.update_traces(textposition="outside", textfont_size=9)
    fig.update_yaxes(range=[0, 1.1])
    fig.update_xaxes(tickangle=-35)
    return _apply_layout(fig, "5-Fold CV Metrics per Model", height=480)


# ── Model Evaluation (Binary Classification) ─────────────────────────────

def model_comparison_chart(results):
    display_metrics = [
        "Balanced Accuracy", "Precision", "Recall (TPR)", "Specificity (TNR)",
        "F1", "F2", "MCC", "ROC-AUC", "PR-AUC (Avg Prec)",
    ]
    rows = []
    for model, metrics in results.items():
        for m in display_metrics:
            if m in metrics:
                rows.append({"Model": model, "Metric": m, "Score": metrics[m]})
    melted = pd.DataFrame(rows)

    fig = px.bar(
        melted, x="Metric", y="Score", color="Model", barmode="group",
        text=melted["Score"].round(3),
        color_discrete_sequence=VIVID_CATEGORICAL,
    )
    fig.update_traces(textposition="outside", textfont_size=9)
    fig.update_yaxes(range=[-0.15, 1.15])
    fig.update_xaxes(tickangle=-30)
    return _apply_layout(fig, "Binary Classification Metrics Comparison", height=520)


def roc_curves_chart(roc_data):
    fig = go.Figure()
    colors = VIVID_CATEGORICAL
    for i, (name, fpr, tpr, auc_val) in enumerate(roc_data):
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, name=f"{name} (AUC={auc_val:.3f})",
            line=dict(color=colors[i % len(colors)], width=2.5),
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], name="Random",
        line=dict(color="gray", width=1, dash="dash"), showlegend=False,
    ))
    fig.update_xaxes(title="False Positive Rate")
    fig.update_yaxes(title="True Positive Rate")
    return _apply_layout(fig, "ROC Curves", height=480)


def pr_curves_chart(pr_data):
    fig = go.Figure()
    colors = VIVID_CATEGORICAL
    for i, (name, prec, rec, ap) in enumerate(pr_data):
        fig.add_trace(go.Scatter(
            x=rec, y=prec, name=f"{name} (AP={ap:.3f})",
            line=dict(color=colors[i % len(colors)], width=2.5),
        ))
    fig.update_xaxes(title="Recall", range=[0, 1.05])
    fig.update_yaxes(title="Precision", range=[0, 1.05])
    return _apply_layout(fig, "Precision-Recall Curves", height=480)


def threshold_analysis_chart(y_true, y_prob, model_name):
    thresholds = np.linspace(0.05, 0.95, 100)
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    records = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if y_pred.sum() == 0 or y_pred.sum() == len(y_pred):
            continue
        from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score
        records.append({
            "Threshold": t,
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall":    recall_score(y_true, y_pred, zero_division=0),
            "F1":        f1_score(y_true, y_pred, zero_division=0),
            "F2":        fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        })

    tdf = pd.DataFrame(records)
    fig = go.Figure()
    for metric, color in [("Precision", COLORS["retained"]), ("Recall", COLORS["churned"]),
                          ("F1", COLORS["accent"]), ("F2", COLORS["purple"])]:
        fig.add_trace(go.Scatter(
            x=tdf["Threshold"], y=tdf[metric], name=metric,
            line=dict(color=color, width=2.5),
        ))
    fig.update_xaxes(title="Classification Threshold")
    fig.update_yaxes(title="Score", range=[0, 1.05])
    return _apply_layout(fig, f"Threshold Analysis — {model_name}", height=420)


def probability_distribution_chart(y_true, y_prob, model_name, opt_threshold=0.5):
    tdf = pd.DataFrame({
        "Probability": y_prob,
        "Actual": ["Churned" if y == 1 else "Retained" for y in y_true],
    })
    fig = px.histogram(
        tdf, x="Probability", color="Actual", nbins=50,
        barmode="overlay", opacity=0.65,
        color_discrete_map={"Churned": COLORS["churned"], "Retained": COLORS["retained"]},
    )
    fig.add_vline(x=opt_threshold, line_dash="dash", line_color="black",
                  annotation_text=f"Threshold={opt_threshold:.2f}")
    fig.update_xaxes(title="Predicted Churn Probability")
    fig.update_yaxes(title="Count")
    return _apply_layout(fig, f"Probability Distribution — {model_name}", height=400)


def threshold_comparison_chart(threshold_info):
    rows = []
    for name, info in threshold_info.items():
        for label, metrics in [("Default (0.5)", info["metrics_default"]),
                               (f"Optimal ({info['optimal']:.2f})", info["metrics_optimal"])]:
            rows.append({
                "Model": name, "Threshold": label,
                "F1": metrics["F1"], "Recall (TPR)": metrics["Recall (TPR)"],
                "Precision": metrics["Precision"],
            })
    tdf = pd.DataFrame(rows)
    melted = tdf.melt(id_vars=["Model", "Threshold"], var_name="Metric", value_name="Score")
    fig = px.bar(
        melted, x="Model", y="Score", color="Threshold", barmode="group",
        facet_col="Metric", text=melted["Score"].round(3),
        color_discrete_sequence=[COLORS["retained"], COLORS["churned"]],
    )
    fig.update_traces(textposition="outside", textfont_size=9)
    fig.update_yaxes(range=[0, 1.15])
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return _apply_layout(fig, "Default vs Optimal Threshold Impact", height=450)


def confusion_matrix_chart(cm, model_name):
    labels = ["Retained", "Churned"]
    fig = px.imshow(
        cm, text_auto=True, x=labels, y=labels,
        color_continuous_scale=[[0, "#f0f4f8"], [0.5, "#76B7B2"], [1, "#364F6B"]],
        aspect="equal", labels=dict(x="Predicted", y="Actual"),
    )
    return _apply_layout(fig, f"Confusion Matrix — {model_name}", height=380, width=420)


def feature_importance_chart(importances, feature_names, model_name):
    imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    imp_df = imp_df.sort_values("Importance", ascending=True).tail(20)

    fig = px.bar(
        imp_df, x="Importance", y="Feature", orientation="h",
        color="Importance",
        color_continuous_scale=[[0, "#76B7B2"], [0.5, "#F28E2C"], [1, "#E15759"]],
        text=imp_df["Importance"].round(4),
    )
    fig.update_traces(textposition="outside", textfont_size=10)
    fig.update_coloraxes(showscale=False)
    return _apply_layout(fig, f"Top 20 Feature Importances — {model_name}", height=550)
