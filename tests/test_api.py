import os
import pytest
from fastapi.testclient import TestClient
from src.main import app


def test_health_check():
    """Ensures health check endpoint reports status and model status."""
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "online"
        assert response.json()["model_loaded"] is True


def test_predict_churn_high_risk():
    """Validates real-time inference for a high-risk month-to-month customer."""
    payload = {
        "tenure": 2,
        "monthly_charges": 95.0,
        "total_charges": 190.0,
        "contract": "Month-to-month",
        "internet_service": "Fiber optic",
        "payment_method": "Electronic check",
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["churn_prediction"] in [0, 1]
        assert 0.0 <= data["churn_probability"] <= 1.0
        assert data["risk_level"] in ["Low", "Medium", "High"]


def test_predict_churn_low_risk():
    """Validates real-time inference for a low-risk two-year contract customer."""
    payload = {
        "tenure": 60,
        "monthly_charges": 25.0,
        "total_charges": 1500.0,
        "contract": "Two year",
        "internet_service": "No",
        "payment_method": "Bank transfer",
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["risk_level"] == "Low"