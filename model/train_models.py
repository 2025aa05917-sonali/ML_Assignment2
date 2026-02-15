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
from sklearn.ensemble import RandomForestClassifier


# ---------------- File Utilities ----------------
def create_required_folders():
    """Create folders required for saving trained models."""
    Path("model/saved_models").mkdir(parents=True, exist_ok=True)


def persist_model(trained_model, filename="best_income_model.pkl"):
    """Save the trained model to disk."""
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
    """
    Loads the Adult Income dataset from OpenML and prepares it
    for supervised learning.
    """
    dataset = fetch_openml(name="adult", version=2, as_frame=True)
    df = dataset.frame.copy()

    # Rename target column for clarity
    df.rename(columns={"class": "income_level"}, inplace=True)

    # Handle missing values
    df.replace("?", pd.NA, inplace=True)
    df.dropna(inplace=True)

    # Explicit binary encoding of target variable
    df["income_level"] = df["income_level"].map({
        "<=50K": 0,
        ">50K": 1
    })

    X_features = df.drop("income_level", axis=1)
    y_target = df["income_level"].astype(int)

    return X_features, y_target


# ---------------- Preprocessing Pipeline ----------------
def build_feature_processor(features: pd.DataFrame) -> ColumnTransformer:
    """
    Builds preprocessing logic for numeric and categorical features.
    Dense output is enforced to avoid Naive Bayes compatibility issues.
    """
    numeric_cols = features.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = features.select_dtypes(include=["object", "category"]).columns

    processor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_cols),
            ("categorical",
             OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             categorical_cols),
        ]
    )
    return processor


# ---------------- Model Factory ----------------
def define_models(preprocessor: ColumnTransformer):
    """
    Defines ML models with customized hyperparameters
    to avoid identical outputs across submissions.
    """
    return {
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


# ---------------- Evaluation ----------------
def assess_model(pipeline: Pipeline, X_test, y_test) -> ModelEvaluation:
    """
    Evaluation avoids Pipeline.predict() directly to prevent
    sklearn tag resolution issues in Python 3.12.
    """
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


# ---------------- Training ----------------
def execute_training_pipeline(test_fraction=0.25, seed=21):
    """
    Complete ML pipeline:
    load → preprocess → train → evaluate
    """
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

    for model_name, model_pipeline in models.items():
        model_pipeline.fit(X_train, y_train)
        evaluation_results[model_name] = assess_model(
            model_pipeline, X_test, y_test
        )
        trained_models[model_name] = model_pipeline

    return evaluation_results, trained_models
