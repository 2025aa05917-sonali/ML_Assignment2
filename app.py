import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

from model.train_models import run_experiment

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="ML Assignment 2", layout="wide")

st.title("Machine Learning Model Evaluation")

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Settings")

test_size = st.sidebar.slider(
    "Test data size",
    0.1, 0.5, 0.2, 0.05
)

seed = st.sidebar.number_input(
    "Random seed",
    value=42
)

# -----------------------------
# Dataset upload
# -----------------------------
st.header("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV file (only test data)",
    type=["csv"]
)

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.write("Uploaded dataset preview:")
    st.dataframe(test_df.head())
else:
    st.write("No file uploaded. Using default test split.")

# -----------------------------
# Run models
# -----------------------------
st.header("Run Experiment")

cols, results, trained_models, (X_test, y_test) = run_experiment(
    test_size=test_size,
    seed=seed
)

# -----------------------------
# Model selection
# -----------------------------
st.header("Select Model")

model_name = st.selectbox(
    "Choose a model",
    list(trained_models.keys())
)

model = trained_models[model_name]

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Metrics
# -----------------------------
st.header("Evaluation Metrics")

st.write("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
st.write("Precision:", round(precision_score(y_test, y_pred), 3))
st.write("Recall:", round(recall_score(y_test, y_pred), 3))
st.write("F1 Score:", round(f1_score(y_test, y_pred), 3))

# -----------------------------
# Confusion Matrix
# -----------------------------
st.header("Confusion Matrix")

fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, ax=ax
)
st.pyplot(fig)

# -----------------------------
# Classification Report
# -----------------------------
st.header("Classification Report")

st.text(classification_report(y_test, y_pred))
