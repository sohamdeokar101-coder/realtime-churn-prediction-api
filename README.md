# 📈 End-to-End Real-Time Churn Prediction API

An end-to-end Machine Learning system that trains an XGBoost classifier on Kaggle Telco Customer Churn data, packages preprocessing and model artifacts into a Scikit-learn Pipeline, and serves low-latency real-time inference via FastAPI.

---

## 📌 Features

- **Automated Ingestion:** Downloads and cleans Kaggle Telco Customer Churn data directly from source.
- **Scikit-learn Pipeline:** Enforces categorical one-hot encoding (`OneHotEncoder`) and numerical scaling (`StandardScaler`) in a single serialized pipeline artifact.
- **XGBoost Classifier:** Achieves **0.8397 ROC-AUC** for churn probability classification.
- **FastAPI Inference Service:** Serves real-time `/predict` endpoints with strict Pydantic input validation and risk-level categorization (Low, Medium, High).
- **Automated Testing:** Covered by unit and API integration tests using `pytest` and `TestClient`.

---

## 🛠️ Tech Stack

- **Framework:** FastAPI, Uvicorn
- **ML Pipeline:** Scikit-learn, XGBoost, Pandas, NumPy
- **Serialization:** Joblib
- **Validation:** Pydantic
- **Testing:** PyTest

---

## 🚀 Quick Start

### 1. Installation & Training

```bash
# Clone the repository
git clone [https://github.com/sohamdeokar101-coder/realtime-churn-prediction-api.git](https://github.com/sohamdeokar101-coder/realtime-churn-prediction-api.git)
cd realtime-churn-prediction-api

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Train model and generate artifact
python -m src.train

💻 Live API Request & Output
Terminal Request (curl)

curl -X POST "[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)" \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 1,
    "monthly_charges": 90.0,
    "total_charges": 90.0,
    "contract": "Month-to-month",
    "internet_service": "Fiber optic",
    "payment_method": "Electronic check"
  }'


JSON Response
JSON
{
  "churn_prediction": 1,
  "churn_probability": 0.826,
  "risk_level": "High"
}
