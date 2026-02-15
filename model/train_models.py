from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


@dataclass
class EvaluationResult:
    metrics: Dict[str, float]
    confusion: np.ndarray
    report: str


# ---------------- Dataset Loader ----------------
def load_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    adult = fetch_openml(name="adult", version=2, as_frame=True)
    df = adult.frame.copy()

    df.rename(columns={"class": "income"}, inplace=True)
    df.replace("?", pd.NA, inplace=True)
    df.dropna(inplace=True)

    le = LabelEncoder()
    for col in df.select_dtypes(include="object"):
        df[col] = le.fit_transform(df[col])

    X = df.drop("income", axis=1)
    y = df["income"]

    return X, y


# ---------------- Models ----------------
def get_models() -> Dict[str, Any]:
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000))
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "kNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7))
        ]),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42
        )
    }

    if _HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
            use_label_encoder=False
        )
    else:
        models["Gradient Boosting"] = GradientBoostingClassifier(random_state=42)

    return models


# ---------------- Evaluation ----------------
def _auc_for(model, X_test, y_test) -> float:
    try:
        if hasattr(model, "predict_proba"):
            return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    except Exception:
        pass
    return float("nan")


def evaluate_model(model, X_test, y_test) -> EvaluationResult:
    y_pred = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "AUC": _auc_for(model, X_test, y_test)
    }

    conf = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return EvaluationResult(metrics, conf, report)


# ---------------- Utilities ----------------
def ensure_dirs():
    Path("data").mkdir(exist_ok=True)
    Path("model/saved").mkdir(parents=True, exist_ok=True)


def make_test_csv(X, y, n_rows=50):
    ensure_dirs()
    idx = np.random.choice(len(X), size=n_rows, replace=False)
    X.iloc[idx].to_csv("data/test_data.csv", index=False)
    pd.concat([X.iloc[idx], y.iloc[idx]], axis=1).to_csv(
        "data/test_data_with_target.csv", index=False
    )


def save_model(model, path="model/saved/best_model.pkl"):
    ensure_dirs()
    joblib.dump(model, path)


# ---------------- Main Runner ----------------
def run_experiment(test_size=0.2, random_state=42):
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    models = get_models()
    results = {}
    fitted = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        res = evaluate_model(model, X_test, y_test)
        results[name] = res
        fitted[name] = model

    make_test_csv(X_test, y_test)
    return X.columns.tolist(), results, fitted, (X_test, y_test)


if __name__ == "__main__":
    cols, results, _, _ = run_experiment()
    for k, v in results.items():
        print(f"\n== {k}")
        print(v.metrics)
