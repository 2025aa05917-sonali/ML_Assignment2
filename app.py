import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from model.train_models import execute_training_pipeline, prepare_adult_income_data

# ---------------- Page Config ----------------
st.set_page_config(page_title="ML Assignment 2", layout="wide")
st.title("Adult Income Classification")
st.caption("Machine Learning Assignment – Model Comparison Dashboard")

# ---------------- Sidebar ----------------
st.sidebar.header("Experiment Settings")

test_size = st.sidebar.slider(
    "Test size",
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

st.sidebar.divider()

# ---------------- Dataset Section ----------------
st.header("Dataset")

with st.expander("Upload test dataset (optional)"):
    uploaded_file = st.file_uploader(
        "Upload CSV file (test data only)",
        type=["csv"]
    )
    if uploaded_file is not None:
        uploaded_df = pd.read_csv(uploaded_file)
        st.write("Uploaded dataset preview:")
        st.dataframe(uploaded_df.head())

# Download Adult Income dataset
X_full, y_full = prepare_adult_income_data()
adult_df = pd.concat([X_full, y_full.rename("income_level")], axis=1)

st.download_button(
    label="Download Adult Income Dataset (CSV)",
    data=adult_df.to_csv(index=False),
    file_name="adult_income_dataset.csv",
    mime="text/csv"
)

# ---------------- Run Experiment ----------------
st.header("Run Experiment")

with st.spinner("Training models and evaluating performance..."):
    results, models = execute_training_pipeline(
        test_fraction=test_size,
        seed=seed
    )

st.success("Training completed successfully")

# ---------------- Metrics Table ----------------
st.header("Model Comparison – Evaluation Metrics")

metrics_table = []

for model_name, evaluation in results.items():
    row = {"Model": model_name}
    for metric, value in evaluation.scores.items():
        row[metric] = round(value, 4)
    metrics_table.append(row)

metrics_df = pd.DataFrame(metrics_table)
metrics_df.set_index("Model", inplace=True)

st.dataframe(metrics_df, use_container_width=True)

# ---------------- Individual Model Analysis ----------------
st.header("Detailed Model Analysis")

model_name = st.selectbox(
    "Select a model for detailed evaluation",
    list(models.keys())
)

evaluation = results[model_name]

# ---- Metrics ----
st.subheader("Evaluation Metrics")
metrics_col1, metrics_col2 = st.columns(2)

with metrics_col1:
    for k, v in evaluation.scores.items():
        st.metric(label=k, value=f"{v:.4f}")

# ---- Confusion Matrix ----
st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
ConfusionMatrixDisplay(evaluation.confusion).plot(ax=ax)
st.pyplot(fig)

# ---- Classification Report ----
st.subheader("Classification Report")
st.text(evaluation.report)
