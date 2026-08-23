import os
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"


def load_and_preprocess_telco_data() -> pd.DataFrame:
    """Downloads Telco Customer Churn dataset and cleans numeric columns."""
    print(f"📥 Fetching Kaggle Telco Churn dataset from URL...")
    df = pd.read_csv(DATA_URL)

    # Clean total charges (handle whitespace strings)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Map target variable 'Churn' (Yes/No -> 1/0)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Standardize column names to snake_case
    df = df.rename(
        columns={
            "tenure": "tenure",
            "MonthlyCharges": "monthly_charges",
            "TotalCharges": "total_charges",
            "Contract": "contract",
            "InternetService": "internet_service",
            "PaymentMethod": "payment_method",
            "Churn": "churn",
        }
    )

    return df[["tenure", "monthly_charges", "total_charges", "contract", "internet_service", "payment_method", "churn"]]


def train_and_save_model():
    """Trains an XGBoost pipeline on Telco Churn data and saves the artifact."""
    df = load_and_preprocess_telco_data()

    X = df.drop(columns=["churn"])
    y = df["churn"]

    numeric_features = ["tenure", "monthly_charges", "total_charges"]
    categorical_features = ["contract", "internet_service", "payment_method"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("⚡ Training XGBoost model...")
    model_pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)[:, 1]

    print("\n--- Model Evaluation ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

    # Save artifact
    os.makedirs("models", exist_ok=True)
    joblib.dump(model_pipeline, "models/churn_pipeline.joblib")
    print("✅ Model pipeline successfully saved to 'models/churn_pipeline.joblib'")


if __name__ == "__main__":
    train_and_save_model()