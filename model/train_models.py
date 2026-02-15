"""
Author: Sonali Chavan
Purpose:
Training and evaluation pipeline for Adult Income classification.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

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
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


# ---------------- Utilities ----------------
def create_required_folders():
    Path("model/saved_models").mkdir(parents=True, exist_ok=True)


def persist_model(model, filename="best_model.pkl"):
    create_required_folders()
    path = Path("model/saved_models") / filename
    joblib.dump(model, path)
    return path


# ---------------- Result Container ----------------
@dataclass
class ModelEvaluation:
    scores: Dict[str, float]
    confusion: np.ndarray
    report: str


# ---------------- Dataset ----------------
def prepare_adult_income_data() -> Tuple[pd.DataFrame, pd.Series]:
    dataset = fetch_openml(name="adult", version=2, as_frame=True)
    df = dataset.frame.copy()

    df.rename(columns={"class": "income_level"}, inplace=True)
    df.replace("?", pd.NA, inplace=True)
    df.dropna(inplace=True)

    df["income_level"] = df["income_level"].map({"<=50K": 0, ">50K": 1})

    X = df.drop("income_level", axis=1)
    y = df["income_level"].astype(int)

    return X, y


# ---------------- Preprocessing ----------------
def build_feature_processor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ]
    )


# ---------------- Models ----------------
def define_models(preprocessor: ColumnTransformer):
    models = {
        "Logistic Regression": Pipeline([
            ("prep", preprocessor),
            ("model", LogisticRegression(max_iter=1200))
        ]),

        "Decision Tree": Pipeline([
            ("prep", preprocessor),
            ("model", DecisionTreeClassifier(max_depth=10, random_state=21))
        ]),

        "KNN": Pipeline([
            ("prep", preprocessor),
            ("model", KNeighborsClassifier(n_neighbors=7))
        ]),

        "Naive Bayes": Pipeline([
            ("prep", preprocessor),
            ("model", GaussianNB())
        ]),

        "Random Forest": Pipeline([
            ("prep", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=250,
                max_depth=14,
                random_state=21
            ))
        ])
    }

    if HAS_XGB:
        models["XGBoost"] = Pipeline([
            ("prep", preprocessor),
            ("model", XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=21
            ))
        ])

    return models


# ---------------- Evaluation ----------------
def assess_model(pipeline: Pipeline, X_test, y_test) -> ModelEvaluation:
    prep = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["model"]

    X_t = prep.transform(X_test)
    y_pred = clf.predict(X_t)

    scores = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    if hasattr(clf, "predict_proba"):
        scores["ROC-AUC"] = roc_auc_score(
            y_test, clf.predict_proba(X_t)[:, 1]
        )

    return ModelEvaluation(
        scores=scores,
        confusion=confusion_matrix(y_test, y_pred),
        report=classification_report(y_test, y_pred)
    )


# ---------------- Main Entry ----------------
def execute_training_pipeline(test_fraction=0.25, seed=21):
    X, y = prepare_adult_income_data()
    processor = build_feature_processor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_fraction,
        random_state=seed,
        stratify=y
    )

    models = define_models(processor)

    results = {}
    trained = {}

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        results[name] = assess_model(pipeline, X_test, y_test)
        trained[name] = pipeline

    return results, trained
