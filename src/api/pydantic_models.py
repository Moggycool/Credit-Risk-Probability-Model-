"""Pydantic models for API request and response validation."""
from typing import Dict, Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    Flexible request model: either provide a mapping of feature name -> value
    (recommended to match training feature names), or provide a raw dict.
    Example JSON body:
    {
      "customer_id": "C123",
      "features": {
        "Amount_sum": 120.5,
        "Amount_mean": 40.2,
        ...
      }
    }
    """
    customer_id: Optional[str] = Field(
        None, description="Optional customer identifier")
    features: Dict[str, float] = Field(
        ..., description="Mapping of feature name to numeric value")


class PredictionResponse(BaseModel):
    """ Response model for prediction results."""
    probability: float = Field(..., ge=0.0, le=1.0,
                               description="Predicted risk probability (0..1)")
    predicted_class: Optional[int] = Field(
        None, description="Predicted class label if available (0/1)")
    customer_id: Optional[str] = Field(
        None, description="Echoed customer id if provided")
    model: Optional[str] = Field(
        None, description="Model identifier used for prediction")
