# Customer Churn Analysis

An interactive Streamlit application that **automatically analyzes any customer churn dataset**. Upload any CSV with a churn/target column and get instant exploratory analysis, 11 machine learning models with binary classification optimization, and AI-powered insights via OpenAI or Google Gemini.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

### Works with Any Churn Dataset
- **Auto-detects** target column (`Churn`, `churned`, `Exited`, `attrition`, `target`, etc.)
- **Auto-detects** and drops ID columns (`customerID`, `RowNumber`, etc.)
- **Auto-classifies** features as categorical or numerical
- **Dynamically builds** all charts, metrics, and insights based on your data

### Exploratory Data Analysis
- **Dashboard** — Auto-generated KPIs, churn distribution, top drivers
- **Data Explorer** — Browse raw data, data types, missing values, and summary statistics
- **Categorical Analysis** — Churn rates by every detected categorical feature
- **Numerical Analysis** — Distributions, box plots, scatter plots for all numerical features
- **Correlation** — Heatmap of feature correlations with the churn target

### Predictive Modeling
- **11 models** organized by MECE taxonomy:
  - *Linear* — Logistic Regression
  - *Bagging Ensembles* — Random Forest, Extra Trees
  - *Boosting Ensembles* — AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost
  - *Kernel / Distance* — SVM (RBF), K-Nearest Neighbors
  - *Neural Networks* — MLP Neural Network
- **Two-phase workflow**: benchmark all models with 5-fold stratified cross-validation, then train the top performers with a full binary-optimized pipeline
- **Class imbalance handling** via `class_weight='balanced'`, `scale_pos_weight`, and `sample_weight`
- **Threshold optimization** using Youden's J statistic
- **10 evaluation metrics**: Accuracy, Balanced Accuracy, Precision, Recall, Specificity, F1, F2, MCC, ROC-AUC, PR-AUC
- Interactive charts: ROC curves, PR curves, confusion matrices, feature importance, probability distributions, threshold analysis

### AI-Powered Insights
- Supports **OpenAI** (GPT-4o-mini) and **Google Gemini** (2.5 Pro)
- Comprehensive retention strategy generation
- Customer segment deep-dive analysis
- Custom question answering with dataset context

## Quick Start

### Prerequisites
- Python 3.9+

### Installation

```bash
git clone https://github.com/chenzhaograce/Customer-Churn-Analysis.git
cd Customer-Churn-Analysis
pip install -r requirements.txt
```

### Run

From the project directory:

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

**Important:** start the app with **`streamlit run app.py`**, not `python app.py`. Running the file with plain Python does not start Streamlit’s server, so the browser will show nothing useful or you will see “missing ScriptRunContext” warnings.

If something fails locally:

1. Run **`python check_env.py`** (uses the same Python you run Streamlit with). It prints which imports are broken.
2. Install deps: **`pip install -r requirements.txt`** (or `python -m pip install -r requirements.txt`).
3. If **`Address already in use`**, pick another port: **`streamlit run app.py --server.port 8502`**.
4. Copy the **full traceback** from the terminal (or the red box in the browser) when asking for help.

### Streamlit Community Cloud

Refreshing the browser **does not** redeploy new code from GitHub or change the server’s Python version. If fixes do not appear:

1. **Confirm the latest commit** is on `main` (the app pulls from GitHub on each run, but a stale container can lag). In [share.streamlit.io](https://share.streamlit.io/) open your app → **⋮ Manage app** → check the build logs for the commit hash.
2. **Reboot the app**: **Manage app** → **⋮** → **Reboot app** (forces a clean container and reinstall).
3. **Python version**: Under **Manage app** → **Settings** → **Advanced settings**, set **Python** to **3.12** if you see errors with PyArrow or `st.dataframe` on **3.14** (Community Cloud may offer the newest Python before all libraries are fully aligned).
4. **Health check / app won’t start**: `requirements.txt` pins **XGBoost to 2.0.x** (`<2.1`). Versions **2.1+** and **3.x** can pull **`nvidia-nccl-cu12`** from PyPI on Linux, which Streamlit Cloud’s **CPU-only** workers do not need and which can crash or reset the process during startup. After changing dependencies, reboot the app so the environment rebuilds.
5. **Clear session cache** (optional): in the running app, menu **☰** → **Clear cache**.

### AI Insights (Optional)

To enable AI-powered insights, enter your API key in the sidebar:
- **OpenAI**: Get a key at [platform.openai.com](https://platform.openai.com/)
- **Gemini**: Get a key at [aistudio.google.com](https://aistudio.google.com/)

## Project Structure

```
├── app.py              # Main Streamlit application
├── config.py           # Color palette, CSS styling
├── data_processor.py   # Auto-detection, loading, preprocessing
├── charts.py           # Plotly visualization functions
├── ml_models.py        # ML model registry, training, evaluation
├── llm_insights.py     # OpenAI & Gemini API integration
├── requirements.txt    # Python dependencies
└── data/
    └── telco_churn.csv  # Sample Telco Customer Churn dataset
```

## Dataset Compatibility

The app works with **any CSV** containing a binary churn/target column. It auto-detects:
- **Target columns** named: `Churn`, `churned`, `Exited`, `attrition`, `target`, `label`, `left`
- **Target values**: `Yes/No`, `1/0`, `True/False`, or `Churned/Retained`
- **ID columns**: automatically detected and dropped

### Included Sample
The bundled dataset is [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 customers, 21 features).

### Tested With
- Telco Customer Churn (telecom)
- [Bank Customer Churn](https://github.com/selva86/datasets/blob/master/Churn_Modelling.csv) (banking, `Exited` column)
- Any subscription/SaaS/streaming churn dataset

## Tech Stack

| Layer | Tools |
|-------|-------|
| Frontend | Streamlit, Plotly, custom CSS |
| Data | Pandas, NumPy |
| ML | scikit-learn, XGBoost, LightGBM, CatBoost |
| AI Insights | OpenAI API, Google Gemini API |

## Acknowledgments

Inspired by [Customer Churn Prediction](https://www.kaggle.com/code/bhartiprasad17/customer-churn-prediction) by Bharti Prasad on Kaggle.
