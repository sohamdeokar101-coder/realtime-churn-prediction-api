import os
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from src.schemas import CustomerData, ChurnPredictionResponse

MODEL_PATH = "models/churn_pipeline.joblib"
pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler to load model pipeline on startup."""
    global pipeline
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model artifact not found at {MODEL_PATH}. Run 'python -m src.train' first.")
    pipeline = joblib.load(MODEL_PATH)
    yield


app = FastAPI(
    title="Real-Time Customer Churn Prediction API",
    description="Production-ready FastAPI service predicting customer churn using an XGBoost Pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def health_check():
    return {"status": "online", "model_loaded": pipeline is not None}


@app.post("/predict", response_model=ChurnPredictionResponse)
def predict_churn(customer: CustomerData):
    """Predicts customer churn probability and risk level."""
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    try:
        # Convert input Pydantic schema to DataFrame matching trained feature names
        input_data = {
            "tenure": customer.tenure,
            "monthly_charges": customer.monthly_charges,
            "total_charges": customer.total_charges,
            "contract": customer.contract,
            "internet_service": customer.internet_service,
            "payment_method": customer.payment_method,
        }
        input_df = pd.DataFrame([input_data])

        churn_prob = float(pipeline.predict_proba(input_df)[0][1])
        prediction = int(churn_prob >= 0.5)

        if churn_prob >= 0.7:
            risk = "High"
        elif churn_prob >= 0.35:
            risk = "Medium"
        else:
            risk = "Low"

        return ChurnPredictionResponse(
            churn_prediction=prediction,
            churn_probability=round(churn_prob, 4),
            risk_level=risk,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")