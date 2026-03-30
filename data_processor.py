import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "telco_churn.csv"


@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        st.error("No dataset found. Please upload a CSV file.")
        return None
    return df


@st.cache_data
def preprocess_data(df):
    df = df.copy()
    df.drop(columns=["customerID"], errors="ignore", inplace=True)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 36, 48, 60, 72],
        labels=["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"],
    )

    df["AvgMonthlySpend"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
    )

    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["NumServices"] = df[service_cols].apply(
        lambda row: (row == "Yes").sum(), axis=1
    )

    return df


@st.cache_data
def get_encoded_data(df):
    df_encoded = df.copy()
    df_encoded.drop(columns=["tenure_group", "AvgMonthlySpend", "NumServices"],
                    errors="ignore", inplace=True)

    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}

    binary_cols = [
        "gender", "Partner", "Dependents", "PhoneService",
        "PaperlessBilling", "Churn", "SeniorCitizen",
    ]
    for col in binary_cols:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map(binary_map).fillna(df_encoded[col])

    multi_cat_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract",
        "PaymentMethod",
    ]
    df_encoded = pd.get_dummies(df_encoded, columns=multi_cat_cols, drop_first=True)

    for col in df_encoded.columns:
        df_encoded[col] = pd.to_numeric(df_encoded[col], errors="coerce")
    df_encoded.dropna(inplace=True)

    return df_encoded


def get_churn_summary(df):
    total = len(df)
    churned = (df["Churn"] == "Yes").sum()
    retained = total - churned
    churn_rate = churned / total * 100

    churned_charges = df[df["Churn"] == "Yes"]["MonthlyCharges"].mean()
    retained_charges = df[df["Churn"] == "No"]["MonthlyCharges"].mean()
    avg_tenure_churned = df[df["Churn"] == "Yes"]["tenure"].mean()
    avg_tenure_retained = df[df["Churn"] == "No"]["tenure"].mean()

    return {
        "total_customers": total,
        "churned": churned,
        "retained": retained,
        "churn_rate": churn_rate,
        "avg_monthly_churned": churned_charges,
        "avg_monthly_retained": retained_charges,
        "avg_tenure_churned": avg_tenure_churned,
        "avg_tenure_retained": avg_tenure_retained,
        "revenue_at_risk": df[df["Churn"] == "Yes"]["MonthlyCharges"].sum(),
    }


def get_categorical_churn_rates(df, col):
    ct = df.groupby([col, "Churn"]).size().unstack(fill_value=0)
    ct["Total"] = ct.sum(axis=1)
    if "Yes" in ct.columns:
        ct["Churn Rate (%)"] = (ct["Yes"] / ct["Total"] * 100).round(1)
    else:
        ct["Churn Rate (%)"] = 0.0
    return ct.reset_index()
