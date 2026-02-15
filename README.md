ML ASSIGNMENT 2 – ADULT INCOME CLASSIFICATION
Author: Sonali Chavan

--------------------------------------------------
a. Problem Statement
--------------------------------------------------
The objective of this project is to build and evaluate multiple machine learning
classification models to predict whether an individual’s annual income exceeds
$50,000 based on demographic and employment-related attributes.

The task is formulated as a binary classification problem using the Adult Income
(Census Income) dataset. The project involves data preprocessing, model training,
performance evaluation, and comparison of different machine learning algorithms
using standard evaluation metrics.

--------------------------------------------------
b. Dataset Description
--------------------------------------------------
Dataset Name: Adult Income Dataset (Census Income)
Source: OpenML (https://www.openml.org)

The dataset contains census data collected from individuals in the United States.
Each record represents a person described by various attributes such as age,
education, occupation, work class, hours worked per week, marital status, and
capital gain/loss.

Target Variable:
- income_level
  - 0 : Income ≤ $50K
  - 1 : Income > $50K

Data Preprocessing Steps:
- Missing values represented by '?' were replaced and removed
- Categorical features were encoded using One-Hot Encoding
- Numerical features were standardized
- Dataset was split into training and test sets using stratified sampling

--------------------------------------------------
c. Models Used and Evaluation Metrics
--------------------------------------------------
Six machine learning models were implemented and evaluated using the same
training and testing data for fair comparison.

Evaluation Metrics Used:
- Accuracy
- ROC-AUC
- Precision
- Recall
- F1 Score

--------------------------------------------------
Model Comparison Table
--------------------------------------------------

ML Model Name                Accuracy   AUC     Precision   Recall   F1
---------------------------------------------------------------------------
Logistic Regression          0.8465     0.9030  0.7291      0.6053   0.6615
Decision Tree                0.8550     0.8982  0.7642      0.6003   0.6724
K-Nearest Neighbors          0.8360     0.8752  0.6857      0.6246   0.6537
Naive Bayes                  0.5874     0.8152  0.3684      0.9308   0.5279
Random Forest (Ensemble)     0.8589     0.9133  0.7911      0.5853   0.6728
XGBoost (Ensemble)           0.8683     0.9253  0.7866      0.6431   0.7076

--------------------------------------------------
d. Observations on Model Performance
--------------------------------------------------

ML Model Name: Logistic Regression
Observation:
Logistic Regression provides a strong baseline performance with good accuracy
and AUC. It performs well on balanced metrics but has lower recall compared to
ensemble models.

ML Model Name: Decision Tree
Observation:
Decision Tree achieves slightly better accuracy than Logistic Regression but is
prone to overfitting. Performance is sensitive to depth and split parameters.

ML Model Name: K-Nearest Neighbors
Observation:
kNN performs reasonably well but is computationally expensive and sensitive to
feature scaling. It shows moderate performance across all metrics.

ML Model Name: Naive Bayes
Observation:
Naive Bayes achieves very high recall, indicating strong ability to detect high-income
cases. However, its low precision leads to many false positives, reducing overall
accuracy.

ML Model Name: Random Forest (Ensemble)
Observation:
Random Forest improves upon single decision trees by reducing overfitting.
It achieves strong accuracy and AUC, making it a robust and reliable model.

ML Model Name: XGBoost (Ensemble)
Observation:
XGBoost delivers the best overall performance across most evaluation metrics.
It effectively captures complex patterns in the data and provides the highest
accuracy, AUC, and F1 score among all models tested.

--------------------------------------------------
End of README
--------------------------------------------------
