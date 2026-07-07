"""
Customer Churn Prediction
--------------------------
Predicts whether a telecom customer will churn (leave the service) using
Logistic Regression and Random Forest, based on the IBM/Kaggle
Telco Customer Churn dataset.

Author: Mubashshera Anjum Sajid Khan
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

DATA_PATH = "telco_churn.csv"


def load_data(path: str) -> pd.DataFrame:
    """Load the raw CSV into a DataFrame."""
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, fix dtypes, drop irrelevant columns."""
    df = df.copy()

    # TotalCharges has blank strings for new customers (tenure=0) -> convert & fill
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # customerID is just an identifier, not predictive
    df = df.drop(columns=["customerID"])

    return df


def encode_features(df: pd.DataFrame):
    """Label-encode the target and one-hot encode categorical features."""
    df = df.copy()

    # Target: Churn (Yes/No) -> 1/0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df


def train_models(X_train, y_train):
    """Train Logistic Regression and Random Forest, return both."""
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)

    return log_reg, rf


def evaluate_model(name, model, X_test, y_test):
    """Print accuracy, precision, recall, F1, and confusion matrix."""
    preds = model.predict(X_test)

    print(f"\n--- {name} ---")
    print(f"Accuracy : {accuracy_score(y_test, preds):.3f}")
    print(f"Precision: {precision_score(y_test, preds):.3f}")
    print(f"Recall   : {recall_score(y_test, preds):.3f}")
    print(f"F1 Score : {f1_score(y_test, preds):.3f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))


def top_churn_drivers(rf_model, feature_names, top_n=10):
    """Print the top N features driving churn, from Random Forest importances."""
    importances = pd.Series(rf_model.feature_importances_, index=feature_names)
    top_features = importances.sort_values(ascending=False).head(top_n)

    print(f"\nTop {top_n} Churn Drivers (Random Forest feature importance):")
    for feat, score in top_features.items():
        print(f"  {feat:<35} {score:.4f}")


def main():
    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = encode_features(df)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale numeric features (helps Logistic Regression converge cleanly)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_reg, rf = train_models(X_train_scaled, y_train)

    evaluate_model("Logistic Regression", log_reg, X_test_scaled, y_test)
    evaluate_model("Random Forest", rf, X_test_scaled, y_test)

    top_churn_drivers(rf, X.columns, top_n=10)


if __name__ == "__main__":
    main()
