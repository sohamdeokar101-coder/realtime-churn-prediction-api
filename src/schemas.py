from pydantic import BaseModel, Field
from typing import Literal


class CustomerData(BaseModel):
    tenure: int = Field(..., ge=0, description="Months customer has stayed with the company")
    monthly_charges: float = Field(..., ge=0.0, description="Amount charged monthly")
    total_charges: float = Field(..., ge=0.0, description="Total amount charged over tenure")
    contract: Literal["Month-to-month", "One year", "Two year"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    payment_method: Literal["Electronic check", "Mailed check", "Bank transfer", "Credit card"]

    model_config = {
        "json_schema_extra": {
            "example": {
                "tenure": 3,
                "monthly_charges": 85.50,
                "total_charges": 256.50,
                "contract": "Month-to-month",
                "internet_service": "Fiber optic",
                "payment_method": "Electronic check"
            }
        }
    }


class ChurnPredictionResponse(BaseModel):
    churn_prediction: int = Field(..., description="0 for Stay, 1 for Churn")
    churn_probability: float = Field(..., description="Probability of customer churning")
    risk_level: str = Field(..., description="Risk categorization: Low, Medium, or High")