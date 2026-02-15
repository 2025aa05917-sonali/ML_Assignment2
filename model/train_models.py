import pandas as pd
import joblib
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Load Adult Income dataset directly
adult = fetch_openml(name="adult", version=2, as_frame=True)

data = adult.frame

# Rename target column for clarity
data.rename(columns={"class": "income"}, inplace=True)

# Drop missing values
data.replace("?", pd.NA, inplace=True)
data.dropna(inplace=True)

# Encode categorical features
le = LabelEncoder()
for col in data.select_dtypes(include="object"):
    data[col] = le.fit_transform(data[col])

X = data.drop("income", axis=1)
y = data["income"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "model/scaler.pkl")

# Models
models = {
    "logistic": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(),
    "knn": KNeighborsClassifier(),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# Train and save models
for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f"model/{name}.pkl")

print("✅ Adult Income models trained successfully (loaded via OpenML)")
