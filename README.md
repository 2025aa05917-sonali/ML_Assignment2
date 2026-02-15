# Adult Income Classification – ML Assignment

## Problem Statement
Predict whether a person earns more than $50K per year using demographic data.

## Dataset
Adult Income dataset loaded dynamically from OpenML using scikit-learn.

## Models Used
- Logistic Regression
- Decision Tree
- k-Nearest Neighbors
- Naive Bayes
- Random Forest
- XGBoost

## How to Run

```bash
pip install -r requirements.txt
python model/train_and_evaluate.py
streamlit run app.py
