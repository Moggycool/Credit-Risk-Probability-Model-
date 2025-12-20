"""Pydantic models for API request and response validation."""

from typing import Dict, Optional, Union, Literal
from pydantic import BaseModel, Field
from datetime import datetime


Numeric = Union[float, int]
RiskCategory = Literal["LOW", "MEDIUM", "HIGH"]


class PredictionRequest(BaseModel):
    """
    Request model for credit risk prediction.
    Model expects exactly these 2 features:
    - Year_mean: Average transaction year (e.g., 2019.0)
    - Month_mean: Average transaction month (e.g., 8.0)

    Example JSON body:
    {
      "customer_id": "C123",
      "features": {
        "Year_mean": 2019.0,
        "Month_mean": 8.0
      }
    }
    """
    customer_id: Optional[str] = Field(
        None,
        description="Optional customer identifier",
        example="CUSTOMER_001"
    )
    features: Dict[str, Numeric] = Field(
        ...,
        description="Mapping of feature name to numeric value. Must include: Year_mean and Month_mean",
        example={"Year_mean": 2019.0, "Month_mean": 8.0}
    )


class PredictionResponse(BaseModel):
    """ Response model for prediction results."""
    probability: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Predicted risk probability (0..1). Values closer to 1 indicate higher risk.",
        example=0.3331
    )
    predicted_class: Optional[int] = Field(
        None,
        description="Predicted class label: 0=Low Risk, 1=High Risk",
        example=0
    )
    risk_category: Optional[RiskCategory] = Field(
        None,
        description="Risk category: LOW (0-33%), MEDIUM (33-66%), HIGH (66-100%)",
        example="MEDIUM"
    )
    risk_score: Optional[int] = Field(
        None,
        ge=0,
        le=100,
        description="Risk score from 0-100 (100 being highest risk)",
        example=33
    )
    recommendation: Optional[str] = Field(
        None,
        description="Business recommendation based on risk assessment",
        example="Review - Moderate risk, consider additional verification"
    )
    customer_id: Optional[str] = Field(
        None,
        description="Echoed customer id if provided",
        example="CUSTOMER_001"
    )
    model: Optional[str] = Field(
        None,
        description="Model identifier used for prediction",
        example="local:models/logistic_champion_fixed.joblib"
    )
    timestamp: Optional[str] = Field(
        None,
        description="Timestamp of prediction",
        example="2024-01-15T10:30:00.000Z"
    )
