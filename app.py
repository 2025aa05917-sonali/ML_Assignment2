import streamlit as st
from model.train_models import (
    execute_training_pipeline,
    persist_model
)

st.set_page_config(page_title="Income Prediction System", layout="wide")

st.title("Adult Income Classification System")
st.markdown(
    """
    This application evaluates multiple machine learning models
    to predict whether an individual's income exceeds $50K/year
    based on demographic and employment attributes.
    """
)

st.sidebar.header("Experiment Settings")
test_fraction = st.sidebar.slider("Test Data Percentage", 0.15, 0.35, 0.25)
random_seed = st.sidebar.number_input("Random Seed", value=21)

st.info("Training models with selected parameters…")

results, trained_models = execute_training_pipeline(
    test_fraction=test_fraction,
    seed=random_seed
)

st.success("Model training completed successfully!")

st.header("Model Performance Comparison")

best_model = None
best_accuracy = 0

for model_name, evaluation in results.items():
    st.subheader(model_name)
    st.json(evaluation.scores)

    if evaluation.scores["Accuracy"] > best_accuracy:
        best_accuracy = evaluation.scores["Accuracy"]
        best_model = trained_models[model_name]

if best_model:
    saved_path = persist_model(best_model)
    st.success(f"Best model saved at: `{saved_path}`")
