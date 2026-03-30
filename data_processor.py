import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

SAMPLE_DATASETS = {
    "Telco Churn (7K customers)": "telco_churn.csv",
    "Bank Churn (10K customers)": "bank_churn.csv",
    "Streaming Churn (15K customers)": "streaming_churn.csv",
    "Netflix Churn (1K customers)": "netflix_large_user_data.xlsx",
    "Spotify Churn (8K customers)": "spotify_churn_dataset.csv",
}

_TARGET_ALIASES = {
    "churn", "churned", "is_churn", "is_churned", "target", "label",
    "attrition", "left", "exited",
}


def _detect_target(df):
    for col in df.columns:
        if col.lower().strip() in _TARGET_ALIASES:
            return col
    for col in df.columns:
        if "churn" in col.lower():
            return col
    for col in df.columns:
        if "exit" in col.lower() or "attrit" in col.lower() or "left" in col.lower():
            return col
    return None


def _detect_id_cols(df):
    id_cols = []
    for col in df.columns:
        low = col.lower().strip().replace(" ", "_")
        if low.endswith("id") or low == "id" or low.endswith("_id"):
            id_cols.append(col)
        elif low in ("rownumber", "row_number", "index", "row", "surname"):
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


def _read_file(path_or_buffer, file_name: str = ""):
    """Read a file based on its extension."""
    name = file_name.lower() if file_name else ""
    if hasattr(path_or_buffer, "name"):
        name = path_or_buffer.name.lower()

    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(path_or_buffer, engine="openpyxl")
    if name.endswith(".json"):
        return pd.read_json(path_or_buffer)
    if name.endswith((".parquet", ".pq")):
        return pd.read_parquet(path_or_buffer)
    return pd.read_csv(path_or_buffer)


@st.cache_data
def load_data(uploaded_file=None, sample_name: str = ""):
    if uploaded_file is not None:
        return _read_file(uploaded_file)

    if sample_name and sample_name in SAMPLE_DATASETS:
        fpath = DATA_DIR / SAMPLE_DATASETS[sample_name]
        if fpath.exists():
            return _read_file(fpath, fpath.name)

    default = DATA_DIR / "telco_churn.csv"
    if default.exists():
        return pd.read_csv(default)

    st.error("No dataset found. Please upload a file or select a sample.")
    return None


@st.cache_data
def load_from_sql(connection_string: str, query: str):
    """Load data from a SQL database."""
    try:
        from sqlalchemy import create_engine
        engine = create_engine(connection_string)
        return pd.read_sql(query, engine)
    except ImportError:
        st.error("sqlalchemy is not installed. Run `pip install sqlalchemy`.")
        return None
    except Exception as e:
        st.error(f"SQL connection error: {e}")
        return None


@st.cache_data
def preprocess_data(df):
    df = df.copy()

    target_col = _detect_target(df)
    if target_col is None:
        st.error("Could not find a churn/target column. Make sure your data has a column "
                 "named 'Churn', 'churned', 'Exited', 'is_churned', 'target', or similar.")
        return None, {}

    if target_col != "Churn":
        df.rename(columns={target_col: "Churn"}, inplace=True)

    if pd.api.types.is_numeric_dtype(df["Churn"]):
        df["Churn"] = df["Churn"].map({1: "Yes", 0: "No"})
    elif pd.api.types.is_bool_dtype(df["Churn"]):
        df["Churn"] = df["Churn"].map({True: "Yes", False: "No"})
    else:
        uniq = set(df["Churn"].dropna().unique())
        if uniq != {"Yes", "No"}:
            pos = {v for v in uniq if str(v).lower() in
                   ("yes", "1", "true", "churned")}
            df["Churn"] = df["Churn"].apply(
                lambda v: "Yes" if v in pos else "No"
            )

    id_cols = _detect_id_cols(df)
    df.drop(columns=id_cols, errors="ignore", inplace=True)

    # Drop date/datetime columns (not useful for ML without engineering)
    date_cols = [c for c in df.columns
                 if pd.api.types.is_datetime64_any_dtype(df[c])
                 or ("date" in c.lower() and df[c].dtype == "object")]
    df.drop(columns=date_cols, errors="ignore", inplace=True)

    cat_cols, num_cols = _classify_columns(df, "Churn", id_cols + date_cols)

    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    for col in cat_cols:
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() == 2:
            vals = sorted(df[col].unique())
            df[col] = df[col].map({vals[0]: "No", vals[1]: "Yes"})

    meta = {
        "target": "Churn",
        "id_cols_dropped": id_cols,
        "date_cols_dropped": date_cols,
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

    tenure_col = None
    for col in num_cols:
        if any(kw in col.lower() for kw in
               ("tenure", "duration", "months", "lifetime", "age",
                "days_since", "account_age", "subscription_length",
                "subscription length")):
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
