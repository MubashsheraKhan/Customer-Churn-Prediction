"""
Customer Churn Prediction - Streamlit Dashboard
------------------------------------------------
Interactive demo: enter a customer's details, get a live churn
prediction plus the probability score, powered by a Random Forest
model trained on the Telco Customer Churn dataset.
"""

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

DATA_PATH = "telco_churn.csv"


@st.cache_resource
def load_and_train():
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df = df.drop(columns=["customerID"])
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X = df_encoded.drop(columns=["Churn"])
    y = df_encoded["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_scaled, y_train)

    return model, scaler, X.columns, df


def build_input_row(user_inputs: dict, feature_columns) -> pd.DataFrame:
    """Turn form inputs into a one-hot-encoded row matching training columns."""
    row = pd.DataFrame([user_inputs])
    row_encoded = pd.get_dummies(row)
    row_encoded = row_encoded.reindex(columns=feature_columns, fill_value=0)
    return row_encoded


st.title("📉 Customer Churn Predictor")
st.caption("Enter a customer's profile to predict their likelihood of churning.")

model, scaler, feature_columns, raw_df = load_and_train()

with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )

    with col2:
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
        total_charges = st.slider("Total Charges ($)", 0.0, 9000.0, 800.0)

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    user_inputs = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": tech_support,
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    input_row = build_input_row(user_inputs, feature_columns)
    input_scaled = scaler.transform(input_row)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ Likely to churn — probability: {probability:.1%}")
    else:
        st.success(f"✅ Likely to stay — churn probability: {probability:.1%}")

    st.progress(min(int(probability * 100), 100))

st.divider()
st.caption("Model: Random Forest | Dataset: Telco Customer Churn (7,043 customers)")
