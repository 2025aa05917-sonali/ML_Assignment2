import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay
from model.train_models import execute_training_pipeline

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="ML Assignment 2", layout="wide")
st.title("Adult Income Classification")

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Settings")

test_size = st.sidebar.slider(
    "Test data fraction",
    0.1, 0.5, 0.25, 0.05
)

seed = st.sidebar.number_input(
    "Random seed",
    value=21
)

# -----------------------------
# Dataset upload
# -----------------------------
st.header("Upload Test Dataset (Optional)")

uploaded_file = st.file_uploader(
    "Upload CSV file (for viewing only)",
    type=["csv"]
)

if uploaded_file is not None:
    uploaded_df = pd.read_csv(uploaded_file)
    st.write("Uploaded dataset preview:")
    st.dataframe(uploaded_df.head())
else:
    st.write("No dataset uploaded. Using default Adult Income dataset.")

# -----------------------------
# Run training pipeline
# -----------------------------
st.header("Model Training and Evaluation")

evaluation_results, trained_models = execute_training_pipeline(
    test_fraction=test_size,
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

selected_eval = evaluation_results[model_name]
selected_model = trained_models[model_name]

# -----------------------------
# Display metrics
# -----------------------------
st.header("Evaluation Metrics")

for metric, value in selected_eval.scores.items():
    st.write(f"{metric}: {value:.3f}")

# -----------------------------
# Confusion matrix
# -----------------------------
st.header("Confusion Matrix")

fig, ax = plt.subplots()
ConfusionMatrixDisplay(
    confusion_matrix=selected_eval.confusion
).plot(ax=ax)

st.pyplot(fig)

# -----------------------------
# Classification report
# -----------------------------
st.header("Classification Report")

st.text(selected_eval.report)
