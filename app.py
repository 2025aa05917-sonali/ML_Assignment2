import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="Adult Income Classification", layout="wide")
st.title("Adult Income Classification – ML Model Comparison")

scaler = joblib.load("model/scaler.pkl")

model_name = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

model_files = {
    "Logistic Regression": "model/logistic.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl"
}

model = joblib.load(model_files[model_name])

uploaded_file = st.file_uploader("Upload CSV (test data only)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    data.dropna(inplace=True)

    for col in data.select_dtypes(include="object"):
        data[col] = data[col].astype("category").cat.codes

    X = scaler.transform(data.drop("income", axis=1))
    y_true = data["income"]

    y_pred = model.predict(X)

    st.subheader("Evaluation Metrics")
    st.text(classification_report(y_true, y_pred))

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_true, y_pred))
