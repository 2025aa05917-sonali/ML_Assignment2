from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# ---------------- Utility ----------------
def ensure_dirs():
    Path("model/saved").mkdir(parents=True, exist_ok=True)


def save_model(model, path="model/saved/best_model.pkl"):
    ensure_dirs()
    joblib.dump(model, path)
    return path


# ---------------- Result container ----------------
@dataclass
class EvaluationResult:
    metrics: Dict[str, float]
    confusion: np.ndarray
    report: str


# ---------------- Load Dataset ----------------
def load_dataset():
    adult = fetch_openml(name="adult", version=2, as_frame=True)
    df = adult.frame.copy()

    df.rename(columns={"class": "income"}, inplace=True)
    df.replace("?", pd.NA, inplace=True)
    df.dropna(inplace=True)

    df["income"] = df["income"].map({"<=50K": 0, ">50K": 1}).astype(int)

    X = df.drop("income", axis=1)
    y = df["income"]

    return X, y


# ---------------- Preprocessing ----------------
def build_preprocessor(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns

    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])


# ---------------- Models ----------------
def get_models(preprocessor):
    return {
        "Logistic Regression": Pipeline([
            ("preprocess", preprocessor),
            ("clf", LogisticRegression(max_iter=1000))
        ]),
        "Decision Tree": Pipeline([
            ("preprocess", preprocessor),
            ("clf", DecisionTreeClassifier(random_state=42))
        ]),
        "kNN": Pipeline([
            ("preprocess", preprocessor),
            ("clf", KNeighborsClassifier(n_neighbors=7))
        ]),
        "Naive Bayes": Pipeline([
            ("preprocess", preprocessor),
            ("clf", GaussianNB())
        ]),
        "Random Forest": Pipeline([
            ("preprocess", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
        ]),
        "Gradient Boosting": Pipeline([
            ("preprocess", preprocessor),
            ("clf", GradientBoostingClassifier(random_state=42))
        ]),
    }


# ---------------- Evaluation ----------------
def evaluate_model(model: Pipeline, X_test, y_test) -> EvaluationResult:
    """
    IMPORTANT:
    We DO NOT call model.predict(X_test)
    This avoids sklearn __sklearn_tags__ crash in Python 3.12
    """

    preprocess = model.named_steps["preprocess"]
    clf = model.named_steps["clf"]

    X_test_transformed = preprocess.transform(X_test)
    y_pred = clf.predict(X_test_transformed)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }

    if hasattr(clf, "predict_proba"):
        metrics["AUC"] = roc_auc_score(
            y_test, clf.predict_proba(X_test_transformed)[:, 1]
        )
    else:
        metrics["AUC"] = np.nan

    conf = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return EvaluationResult(metrics, conf, report)


# ---------------- Run Experiment ----------------
def run_experiment(test_size=0.2, random_state=42):
    X, y = load_dataset()
    preprocessor = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    models = get_models(preprocessor)

    results = {}
    fitted = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = evaluate_model(model, X_test, y_test)
        fitted[name] = model

    return X.columns.tolist(), results, fitted, (X_test, y_test)
