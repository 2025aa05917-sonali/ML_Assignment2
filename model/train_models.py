"""
Author: Sonali Chavan
Purpose:
This module handles data loading, preprocessing, model training,
and evaluation for the Adult Income classification task.
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
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

# ---------------- Optional XGBoost ----------------
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ---------------- File Utilities ----------------
def create_required_folders():
    Path("model/saved_models").mkdir(parents=True, exist_ok=True)


def persist_model(trained_model, filename="best_income_model.pkl"):
    create_required_folders()
    path = Path("model/saved_models") / filename
    joblib.dump(trained_model, path)
    return path


# ---------------- Evaluation Result Container ----------------
@dataclass
class ModelEvaluation:
    scores: Dict[str, float]
    confusion: np.ndarray
    report: str


# ---------------- Dataset Loader ----------------
def prepare_adult_income_data() -> Tuple[pd.DataFrame, pd.Series]:
    dataset = fetch_openml(name="adult", version=2, as_frame=True)
    df = dataset.frame.copy()

    df.rename(columns={"class": "income_level"}, inplace=True)
    df.replace("?", pd.NA, inplace=True)
    df.dropna(inplace=True)

    df["income_level"] = df["income_level"].map({
        "<=50K": 0,
        ">50K": 1
    })

    X = df.drop("income_level", axis=1)
    y = df["income_level"].astype(int)

    return X, y


# ---------------- Preprocessing ----------------
def build_feature_processor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = features.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = features.select_dtypes(include=["object", "category"]).columns

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ]
    )


# ---------------- Model Factory ----------------
def define_models(preprocessor: ColumnTransformer):
    models = {
        "Logistic Regression": Pipeline([
            ("prep", preprocessor),
            ("model", LogisticRegression(max_iter=1200, C=0.8))
        ]),

        "Decision Tree": Pipeline([
            ("prep", preprocessor),
            ("model", DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=6,
                random_state=21
            ))
        ]),

        "K-Nearest Neighbors": Pipeline([
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
                n_estimators=280,
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
    else:
        models["Gradient Boosting"] = Pipeline([
            ("prep", preprocessor),
            ("model", GradientBoostingClassifier(random_state=21))
        ])

    return models


# ---------------- Evaluation ----------------
def assess_model(pipeline: Pipeline, X_test, y_test) -> ModelEvaluation:
    transformer = pipeline.named_steps["prep"]
    classifier = pipeline.named_steps["model"]

    X_test_transformed = transformer.transform(X_test)
    predictions = classifier.predict(X_test_transformed)

    scores = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1 Score": f1_score(y_test, predictions)
    }

    if hasattr(classifier, "predict_proba"):
        scores["ROC-AUC"] = roc_auc_score(
            y_test, classifier.predict_proba(X_test_transformed)[:, 1]
        )

    return ModelEvaluation(
        scores=scores,
        confusion=confusion_matrix(y_test, predictions),
        report=classification_report(y_test, predictions)
    )


# ---------------- Training Pipeline ----------------
def execute_training_pipeline(test_fraction=0.25, seed=21):
    X, y = prepare_adult_income_data()
    processor = build_feature_processor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_fraction,
        random_state=seed,
        stratify=y
    )

    models = define_models(processor)

    evaluation_results = {}
    trained_models = {}

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        evaluation_results[name] = assess_model(pipeline, X_test, y_test)
        trained_models[name] = pipeline

    return evaluation_results, trained_models


# ---------------- PRINT OUTPUTS (For Jupyter / README) ----------------
if __name__ == "__main__":
    print("\nRunning Adult Income Classification Experiment")
    print("=" * 60)

    results, _ = execute_training_pipeline()

    for model_name, evaluation in results.items():
        print(f"\nModel: {model_name}")
        print("-" * 60)

        print("Metrics:")
        for metric, value in evaluation.scores.items():
            print(f"  {metric}: {value:.4f}")

        print("\nConfusion Matrix:")
        print(evaluation.confusion)

        print("\nClassification Report:")
        print(evaluation.report)

    print("\nExperiment completed successfully.")
