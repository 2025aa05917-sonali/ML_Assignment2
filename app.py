import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from model.train_models import execute_training_pipeline, prepare_adult_income_data


# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Adult Income Classification",
    layout="wide"
)

st.title("Adult Income Classification")
st.caption("Machine Learning Assignment – Model Comparison & Evaluation")


# ---------------- Sidebar ----------------
st.sidebar.header("Experiment Settings")

test_size = st.sidebar.slider(
    "Test size (hold-out data)",
    min_value=0.1,
    max_value=0.5,
    value=0.25,
    step=0.05
)

seed = st.sidebar.number_input(
    "Random seed",
    value=21,
    step=1
)

st.sidebar.markdown("---")
st.sidebar.subheader("Dataset")

# Load dataset once for download
X_data, y_data = prepare_adult_income_data()
adult_df = X_data.copy()
adult_df["income_level"] = y_data

csv_data = adult_df.to_csv(index=False).encode("utf-8")

st.sidebar.download_button(
    label="Download Adult Income Dataset (CSV)",
    data=csv_data,
    file_name="adult_income.csv",
    mime="text/csv"
)


# ---------------- Run Experiment ----------------
st.header("Run Experiment")

with st.spinner("Training models and evaluating performance..."):
    results, models = execute_training_pipeline(
        test_fraction=test_size,
        seed=seed
    )

st.success("Training completed successfully!")


# ---------------- Model Selection ----------------
st.header("Model Selection")

model_name = st.selectbox(
    "Choose a model to view results",
    list(models.keys())
)

evaluation = results[model_name]


# ---------------- Metrics Display ----------------
st.header("Evaluation Metrics")

metric_cols = st.columns(len(evaluation.scores))

for col, (metric, value) in zip(metric_cols, evaluation.scores.items()):
    col.metric(label=metric, value=f"{value:.3f}")


# ---------------- Confusion Matrix ----------------
st.header("Confusion Matrix")

fig, ax = plt.subplots()
ConfusionMatrixDisplay(evaluation.confusion).plot(ax=ax, colorbar=False)
st.pyplot(fig)


# ---------------- Classification Report ----------------
st.header("Classification Report")

st.code(evaluation.report, language="text")


# ---------------- Footer ----------------
st.markdown("---")
st.caption(
    "Models included: Logistic Regression, Decision Tree, KNN, Naive Bayes, "
    "Random Forest, XGBoost / Gradient Boosting"
)
