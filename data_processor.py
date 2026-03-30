import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "telco_churn.csv"

_TARGET_ALIASES = {"churn", "churned", "is_churn", "target", "label", "attrition", "left", "exited"}


def _detect_target(df):
    for col in df.columns:
        if col.lower().strip() in _TARGET_ALIASES:
            return col
    for col in df.columns:
        if "churn" in col.lower():
            return col
    return None


def _detect_id_cols(df):
    id_cols = []
    for col in df.columns:
        low = col.lower().strip()
        if low.endswith("id") or low == "id" or low.endswith("_id"):
            id_cols.append(col)
        elif low in ("rownumber", "row_number", "index", "row"):
            id_cols.append(col)
        elif df[col].nunique() == len(df) and df[col].dtype == "object":
            id_cols.append(col)
    return id_cols


def _classify_columns(df, target_col, id_cols):
    cat_cols, num_cols = [], []
    skip = {target_col} | set(id_cols)
    for col in df.columns:
        if col in skip:
            continue
        if df[col].dtype in ("object", "category", "bool"):
            cat_cols.append(col)
        elif df[col].nunique() <= 10:
            cat_cols.append(col)
        else:
            num_cols.append(col)
    return cat_cols, num_cols


@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    st.error("No dataset found. Please upload a CSV file.")
    return None


@st.cache_data
def preprocess_data(df):
    df = df.copy()

    target_col = _detect_target(df)
    if target_col is None:
        st.error("Could not find a churn/target column. Make sure your CSV has a column "
                 "named 'Churn', 'churned', 'target', 'attrition', or similar.")
        return None, {}

    # Standardise target → "Churn" with Yes / No
    if target_col != "Churn":
        df.rename(columns={target_col: "Churn"}, inplace=True)

    if pd.api.types.is_numeric_dtype(df["Churn"]):
        df["Churn"] = df["Churn"].map({1: "Yes", 0: "No"})
    elif pd.api.types.is_bool_dtype(df["Churn"]):
        df["Churn"] = df["Churn"].map({True: "Yes", False: "No"})
    else:
        uniq = set(df["Churn"].dropna().unique())
        if uniq != {"Yes", "No"}:
            pos = {v for v in uniq if str(v).lower() in ("yes", "1", "true", "churned")}
            df["Churn"] = df["Churn"].apply(lambda v: "Yes" if v in pos else "No")

    id_cols = _detect_id_cols(df)
    df.drop(columns=id_cols, errors="ignore", inplace=True)

    cat_cols, num_cols = _classify_columns(df, "Churn", id_cols)

    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    # Convert numeric-looking categoricals (e.g. SeniorCitizen 0/1)
    for col in cat_cols:
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() == 2:
            vals = sorted(df[col].unique())
            df[col] = df[col].map({vals[0]: "No", vals[1]: "Yes"})

    meta = {
        "target": "Churn",
        "id_cols_dropped": id_cols,
        "categorical_cols": [c for c in cat_cols if c in df.columns],
        "numeric_cols": [c for c in num_cols if c in df.columns],
    }
    return df, meta


@st.cache_data
def get_encoded_data(df, cat_cols, num_cols):
    df_enc = df.copy()

    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0,
                  "True": 1, "False": 0, "true": 1, "false": 0}

    binary_cols = [c for c in ["Churn"] + cat_cols
                   if c in df_enc.columns and df_enc[c].nunique() <= 2]
    multi_cat = [c for c in cat_cols
                 if c in df_enc.columns and c not in binary_cols]

    for col in binary_cols:
        mapped = df_enc[col].map(binary_map)
        if mapped.isnull().all():
            vals = df_enc[col].dropna().unique()
            if len(vals) == 2:
                mapped = df_enc[col].map({vals[0]: 0, vals[1]: 1})
        df_enc[col] = mapped.fillna(df_enc[col])

    if multi_cat:
        df_enc = pd.get_dummies(df_enc, columns=multi_cat, drop_first=True)

    for col in df_enc.columns:
        df_enc[col] = pd.to_numeric(df_enc[col], errors="coerce")
    df_enc.dropna(inplace=True)

    return df_enc


def get_churn_summary(df, num_cols):
    total = len(df)
    churned = int((df["Churn"] == "Yes").sum())
    retained = total - churned
    churn_rate = churned / total * 100 if total else 0

    summary = {
        "total_customers": total,
        "churned": churned,
        "retained": retained,
        "churn_rate": churn_rate,
    }

    # Find a monetary / key numeric column for "revenue at risk"
    money_col = None
    for col in num_cols:
        if any(kw in col.lower() for kw in
               ("charge", "revenue", "price", "spend", "bill", "fee",
                "amount", "payment", "cost", "income", "salary")):
            money_col = col
            break
    if money_col is None and num_cols:
        money_col = num_cols[0]

    if money_col:
        summary["money_col"] = money_col
        summary["avg_money_churned"] = df[df["Churn"] == "Yes"][money_col].mean()
        summary["avg_money_retained"] = df[df["Churn"] == "No"][money_col].mean()
        summary["revenue_at_risk"] = df[df["Churn"] == "Yes"][money_col].sum()

    # Find a tenure / duration column
    tenure_col = None
    for col in num_cols:
        if any(kw in col.lower() for kw in
               ("tenure", "duration", "months", "lifetime", "age",
                "days_since", "account_age", "subscription_length")):
            tenure_col = col
            break

    if tenure_col:
        summary["tenure_col"] = tenure_col
        summary["avg_tenure_churned"] = df[df["Churn"] == "Yes"][tenure_col].mean()
        summary["avg_tenure_retained"] = df[df["Churn"] == "No"][tenure_col].mean()

    return summary


def get_categorical_churn_rates(df, col):
    ct = df.groupby([col, "Churn"]).size().unstack(fill_value=0)
    ct["Total"] = ct.sum(axis=1)
    if "Yes" in ct.columns:
        ct["Churn Rate (%)"] = (ct["Yes"] / ct["Total"] * 100).round(1)
    else:
        ct["Churn Rate (%)"] = 0.0
    return ct.reset_index()
