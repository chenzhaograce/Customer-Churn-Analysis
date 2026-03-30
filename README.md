# Customer Churn Analysis

An interactive Streamlit application for analyzing and predicting telecom customer churn. Features automated exploratory data analysis, 11 machine learning models with binary classification optimization, and AI-powered insights via OpenAI or Google Gemini.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

### Exploratory Data Analysis
- **Dashboard** — Key metrics, churn distribution, tenure and charge analysis at a glance
- **Data Explorer** — Browse raw data, data types, missing values, and summary statistics
- **Demographics** — Churn patterns by gender, senior citizen status, partner, and dependents
- **Services** — Impact of internet service type, add-ons (security, backup, tech support), and streaming services
- **Billing & Contract** — Contract type, payment method, monthly/total charges vs. churn
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

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### AI Insights (Optional)

To enable AI-powered insights, enter your API key in the sidebar:
- **OpenAI**: Get a key at [platform.openai.com](https://platform.openai.com/)
- **Gemini**: Get a key at [aistudio.google.com](https://aistudio.google.com/)

## Project Structure

```
├── app.py              # Main Streamlit application
├── config.py           # Color palette, CSS, column definitions
├── data_processor.py   # Data loading, preprocessing, feature engineering
├── charts.py           # Plotly visualization functions
├── ml_models.py        # ML model registry, training, evaluation
├── llm_insights.py     # OpenAI & Gemini API integration
├── requirements.txt    # Python dependencies
└── data/
    └── telco_churn.csv  # Sample Telco Customer Churn dataset
```

## Dataset

The included sample dataset is the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset from Kaggle, containing 7,043 customer records with 21 features including demographics, services, account information, and churn status.

You can also upload your own CSV via the sidebar — the app expects a `Churn` column with `Yes`/`No` values.

## Tech Stack

| Layer | Tools |
|-------|-------|
| Frontend | Streamlit, Plotly, custom CSS |
| Data | Pandas, NumPy |
| ML | scikit-learn, XGBoost, LightGBM, CatBoost |
| AI Insights | OpenAI API, Google Gemini API |

## Acknowledgments

Inspired by [Customer Churn Prediction](https://www.kaggle.com/code/bhartiprasad17/customer-churn-prediction) by Bharti Prasad on Kaggle.
