import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from model.train_models import execute_training_pipeline

st.set_page_config(page_title="ML Assignment 2", layout="wide")
st.title("Adult Income Classification")

st.sidebar.header("Settings")
test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.25, 0.05)
seed = st.sidebar.number_input("Random seed", value=21)

st.header("Run Experiment")

results, models = execute_training_pipeline(
    test_fraction=test_size,
    seed=seed
)

st.header("Select Model")
model_name = st.selectbox("Choose model", list(models.keys()))

evaluation = results[model_name]

st.header("Evaluation Metrics")
for k, v in evaluation.scores.items():
    st.write(f"{k}: {v:.3f}")

st.header("Confusion Matrix")
fig, ax = plt.subplots()
ConfusionMatrixDisplay(evaluation.confusion).plot(ax=ax)
st.pyplot(fig)

st.header("Classification Report")
st.text(evaluation.report)
