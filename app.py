import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from model.train_models import (
    run_experiment, load_dataset, evaluate_model,
    ensure_dirs, save_model
)

st.set_page_config(page_title="Adult Income Classification", layout="wide")
st.title("Adult Income Classification – ML Assignment")

st.markdown(
    "This application trains multiple ML models on the **Adult Income dataset** "
    "loaded dynamically via **OpenML**, compares performance, and allows prediction."
)

# Sidebar
st.sidebar.header("Configuration")
test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
seed = st.sidebar.number_input("Random Seed", value=42)

# Load data preview
X, y = load_dataset()
st.subheader("Dataset Preview")
st.dataframe(X.head())
st.caption(f"Samples: {X.shape[0]} | Features: {X.shape[1]}")

# Train models
cols, results, fitted, (X_test, y_test) = run_experiment(test_size, seed)

# Metrics table
st.subheader("Model Comparison")
rows = []
for name, res in results.items():
    row = {"Model": name}
    row.update(res.metrics)
    rows.append(row)

df = pd.DataFrame(rows)
st.dataframe(df)

# Best model
best_model_name = df.sort_values("AUC", ascending=False).iloc[0]["Model"]
st.success(f"Best Model (by AUC): {best_model_name}")

# Confusion Matrix
res = results[best_model_name]
fig, ax = plt.subplots()
sns.heatmap(res.confusion, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

st.text(res.report)

# Save model
ensure_dirs()
save_model(fitted[best_model_name])
st.sidebar.success("Best model saved as best_model.pkl")
