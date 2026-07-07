# Customer Churn Prediction 📉

A Machine Learning project that predicts whether a telecom customer is likely to churn (cancel their subscription), based on their account and usage profile. Includes an interactive Streamlit dashboard for live predictions.

## 🚀 Live Application
🔗 [Live Demo](https://customer-churn-prediction-jhjwnabyvbhzgnj6eczsh3.streamlit.app/)

## 🧠 How It Works

1. **Data Cleaning**: Handles missing values (e.g. blank `TotalCharges` for new customers) and drops non-predictive identifiers.
2. **Feature Encoding**: Converts categorical fields (contract type, payment method, internet service, etc.) into numerical features via one-hot encoding.
3. **Model Training**: Trains both Logistic Regression and Random Forest classifiers on an 80/20 train-test split.
4. **Evaluation**: Reports accuracy, precision, recall, F1 score, and a confusion matrix for both models.
5. **Feature Importance**: Surfaces the top drivers of churn (e.g. tenure, monthly charges, contract type) using Random Forest importances.

## 📊 Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 80.7% | 65.8% | 56.7% | 60.9% |
| Random Forest | 79.2% | 63.9% | 49.7% | 55.9% |

**Top churn drivers**: Total charges, tenure, monthly charges, fiber optic internet, and month-to-month contracts.

## 🛠️ Tech Stack

- **Language**: Python
- **Machine Learning**: Scikit-Learn (`LogisticRegression`, `RandomForestClassifier`)
- **Data Handling**: Pandas, NumPy
- **Dashboard**: Streamlit
- **Dataset**: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle/IBM)

## 📂 Project Structure

- `churn_prediction.py` — Core ML pipeline: data cleaning, encoding, training, evaluation, feature importance.
- `app.py` — Interactive dashboard for live churn prediction on custom customer inputs.
- `telco_churn.csv` — Dataset (download from Kaggle link above if not included).
- `requirements.txt` — Python package dependencies.
- `README.md` — Project documentation.

## 👩‍💻 Author
Mubashshera Khan
