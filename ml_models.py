import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    fbeta_score, roc_auc_score, roc_curve, confusion_matrix,
    classification_report, balanced_accuracy_score,
    matthews_corrcoef, average_precision_score,
    precision_recall_curve,
)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


# ── MECE Model Taxonomy ───────────────────────────────────────────────────

MODEL_CATEGORIES = {
    "Linear":             ["Logistic Regression"],
    "Bagging Ensembles":  ["Random Forest", "Extra Trees"],
    "Boosting Ensembles": [
        "AdaBoost", "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost",
    ],
    "Kernel / Distance":  ["SVM (RBF)", "K-Nearest Neighbors"],
    "Neural Networks":    ["MLP Neural Network"],
}

# Models that accept sample_weight in fit() but lack a class_weight constructor arg
_NEEDS_SAMPLE_WEIGHT = {"AdaBoost", "Gradient Boosting"}


def _build_registry(scale_pos_weight: float):
    """Build model registry with class-imbalance handling baked in.

    Args:
        scale_pos_weight: n_negative / n_positive (e.g. ~2.77 for 26.5% churn).
    """
    registry = {
        # ── Linear ─────────────────────────────────────────────────────
        "Logistic Regression": lambda: LogisticRegression(
            max_iter=1000, C=1.0, class_weight="balanced", random_state=42,
        ),

        # ── Bagging Ensembles ──────────────────────────────────────────
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "Extra Trees": lambda: ExtraTreesClassifier(
            n_estimators=200, max_depth=12, min_samples_split=5,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),

        # ── Boosting Ensembles ─────────────────────────────────────────
        "AdaBoost": lambda: AdaBoostClassifier(
            n_estimators=150, learning_rate=0.1, random_state=42,
        ),
        "Gradient Boosting": lambda: GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42,
        ),

        # ── Kernel / Distance ──────────────────────────────────────────
        "SVM (RBF)": lambda: SVC(
            kernel="rbf", C=1.0, gamma="scale",
            class_weight="balanced", probability=True, random_state=42,
        ),
        "K-Nearest Neighbors": lambda: KNeighborsClassifier(
            n_neighbors=11, weights="distance", metric="minkowski",
            n_jobs=-1,
        ),

        # ── Neural Networks ────────────────────────────────────────────
        "MLP Neural Network": lambda: MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation="relu",
            solver="adam", max_iter=300, early_stopping=True,
            validation_fraction=0.15, random_state=42,
        ),
    }

    if HAS_XGBOOST:
        spw = scale_pos_weight
        registry["XGBoost"] = lambda: XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw,
            random_state=42, eval_metric="logloss", verbosity=0,
        )

    if HAS_LIGHTGBM:
        spw = scale_pos_weight
        registry["LightGBM"] = lambda: LGBMClassifier(
            n_estimators=200, max_depth=7, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            is_unbalance=True,
            random_state=42, verbosity=-1, force_col_wise=True,
        )

    if HAS_CATBOOST:
        registry["CatBoost"] = lambda: CatBoostClassifier(
            iterations=200, depth=6, learning_rate=0.1,
            auto_class_weights="Balanced",
            random_state=42, verbose=0,
        )

    return registry


# Lightweight registry for get_available_models (no weight info needed)
_STATIC_REGISTRY_NAMES = set()


def _ensure_static_names():
    global _STATIC_REGISTRY_NAMES
    if not _STATIC_REGISTRY_NAMES:
        _STATIC_REGISTRY_NAMES = set(_build_registry(1.0).keys())


def get_category_for_model(name):
    for cat, models in MODEL_CATEGORIES.items():
        if name in models:
            return cat
    return "Other"


def prepare_data(df_encoded, test_size=0.2):
    if "Churn" not in df_encoded.columns:
        return None
    X = df_encoded.drop(columns=["Churn"])
    y = df_encoded["Churn"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index,
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()


def _find_optimal_threshold(y_true, y_prob):
    """Find threshold maximising Youden's J = TPR - FPR."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx])


def _binary_metrics(y_true, y_pred, y_prob):
    """Comprehensive binary-classification metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "Accuracy":          accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision":         precision_score(y_true, y_pred, zero_division=0),
        "Recall (TPR)":      recall_score(y_true, y_pred, zero_division=0),
        "Specificity (TNR)": specificity,
        "F1":                f1_score(y_true, y_pred, zero_division=0),
        "F2":                fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "MCC":               matthews_corrcoef(y_true, y_pred),
        "ROC-AUC":           roc_auc_score(y_true, y_prob),
        "PR-AUC (Avg Prec)": average_precision_score(y_true, y_prob),
    }


@st.cache_data
def train_and_evaluate(df_encoded_json, selected_models, test_size=0.2):
    df_encoded = pd.read_json(pd.io.common.StringIO(df_encoded_json))
    split = prepare_data(df_encoded, test_size)
    if split is None:
        return {}, [], [], {}, {}, [], {}, {}

    X_train, X_test, y_train, y_test, scaler, feature_names = split

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    sample_weights = compute_sample_weight("balanced", y_train)

    registry = _build_registry(scale_pos_weight)

    results = {}
    roc_data = []
    pr_data = []
    confusion_matrices = {}
    feature_importances = {}
    reports = {}
    threshold_info = {}

    for name in selected_models:
        if name not in registry:
            continue

        model = registry[name]()

        if name in _NEEDS_SAMPLE_WEIGHT:
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]

        # default threshold = 0.5
        y_pred_default = (y_prob >= 0.5).astype(int)

        # optimal threshold via Youden's J
        opt_thresh = _find_optimal_threshold(y_test, y_prob)
        y_pred_optimal = (y_prob >= opt_thresh).astype(int)

        results[name] = _binary_metrics(y_test, y_pred_optimal, y_prob)
        results[name]["Optimal Threshold"] = opt_thresh

        threshold_info[name] = {
            "optimal": opt_thresh,
            "metrics_default": _binary_metrics(y_test, y_pred_default, y_prob),
            "metrics_optimal": _binary_metrics(y_test, y_pred_optimal, y_prob),
            "y_prob": y_prob.tolist(),
            "y_true": y_test.tolist(),
        }

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data.append((name, fpr.tolist(), tpr.tolist(), results[name]["ROC-AUC"]))

        # Precision-Recall curve
        prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob)
        pr_data.append((name, prec_arr.tolist(), rec_arr.tolist(), results[name]["PR-AUC (Avg Prec)"]))

        confusion_matrices[name] = confusion_matrix(y_test, y_pred_optimal).tolist()
        reports[name] = classification_report(y_test, y_pred_optimal, output_dict=True)

        if hasattr(model, "feature_importances_"):
            feature_importances[name] = model.feature_importances_.tolist()
        elif hasattr(model, "coef_"):
            coefs = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
            feature_importances[name] = np.abs(coefs).tolist()

    return (results, roc_data, pr_data, confusion_matrices,
            feature_importances, feature_names, reports, threshold_info)


DEFAULT_TOP_5 = [
    "Random Forest", "Logistic Regression", "CatBoost", "Extra Trees", "XGBoost",
]


@st.cache_data
def benchmark_models(df_encoded_json, n_folds=5):
    """Quick stratified k-fold CV to rank all available models.

    Returns a DataFrame with ROC-AUC, F1, PR-AUC, Composite score per model.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    import time

    df_encoded = pd.read_json(pd.io.common.StringIO(df_encoded_json))
    if "Churn" not in df_encoded.columns:
        return pd.DataFrame()

    X = df_encoded.drop(columns=["Churn"])
    y = df_encoded["Churn"].astype(int)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    spw = n_neg / n_pos if n_pos > 0 else 1.0

    registry = _build_registry(spw)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    rows = []
    for name in registry:
        model = registry[name]()
        t0 = time.time()
        roc = cross_val_score(model, X_scaled, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        f1 = cross_val_score(model, X_scaled, y, cv=cv, scoring="f1", n_jobs=-1)
        pr = cross_val_score(model, X_scaled, y, cv=cv, scoring="average_precision", n_jobs=-1)
        elapsed = time.time() - t0
        rows.append({
            "Model": name,
            "Category": get_category_for_model(name),
            "ROC-AUC": np.mean(roc),
            "F1": np.mean(f1),
            "PR-AUC": np.mean(pr),
            "Time (s)": round(elapsed, 1),
        })

    df_results = pd.DataFrame(rows)
    df_results["Composite"] = (
        df_results["ROC-AUC"] + df_results["F1"] + df_results["PR-AUC"]
    ) / 3
    df_results["Rank"] = df_results["Composite"].rank(ascending=False).astype(int)
    return df_results.sort_values("Rank")


def get_available_models():
    """Return models ordered by MECE category."""
    _ensure_static_names()
    ordered = []
    for cat, names in MODEL_CATEGORIES.items():
        for name in names:
            if name in _STATIC_REGISTRY_NAMES:
                ordered.append(name)
    for name in _STATIC_REGISTRY_NAMES:
        if name not in ordered:
            ordered.append(name)
    return ordered


def get_available_models_with_categories():
    _ensure_static_names()
    out = []
    for cat, names in MODEL_CATEGORIES.items():
        for name in names:
            if name in _STATIC_REGISTRY_NAMES:
                out.append((cat, name))
    return out
