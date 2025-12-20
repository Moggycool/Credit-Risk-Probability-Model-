"""Pydantic models for API request and response validation."""

from typing import Dict, Optional, Union, Literal
from pydantic import BaseModel, Field
from datetime import datetime


Numeric = Union[float, int]
RiskCategory = Literal["LOW", "MEDIUM", "HIGH"]


class PredictionRequest(BaseModel):
    """
    Request model for credit risk prediction.

    The model expects exactly 2 features (based on training):
    - Year_mean: Average transaction year (float, e.g., 2019.0)
    - Month_mean: Average transaction month (float, e.g., 8.0)

    These features were selected during model training as the most predictive.

    Example JSON body:
    {
      "customer_id": "CUST_001",
      "features": {
        "Year_mean": 2019.0,
        "Month_mean": 8.0
      }
    }
    """
    customer_id: Optional[str] = Field(
        None,
        description="Optional customer identifier for tracking",
        example="CUSTOMER_001",
        json_schema_extra={
            "example": "CUST_2024_001"
        }
    )
    features: Dict[str, Numeric] = Field(
        ...,
        description="Mapping of feature names to numeric values. "
        "Must include: Year_mean and Month_mean",
        example={"Year_mean": 2019.0, "Month_mean": 8.0},
        json_schema_extra={
            "example": {
                "Year_mean": 2019.0,
                "Month_mean": 8.0
            }
        }
    )

    class Config:
        schema_extra = {
            "example": {
                "customer_id": "CUST_001",
                "features": {
                    "Year_mean": 2019.0,
                    "Month_mean": 8.0
                }
            }
        }


class PredictionResponse(BaseModel):
    """ 
    Response model for prediction results.

    Provides comprehensive risk assessment including probability,
    classification, risk category, and business recommendation.
    """
    probability: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Predicted risk probability (0..1). "
        "Higher values indicate higher risk of default.",
        example=0.6669,
        json_schema_extra={
            "example": 0.6669
        }
    )
    predicted_class: Optional[int] = Field(
        None,
        description="Binary classification: 0 = Low Risk, 1 = High Risk. "
        "Threshold is 0.5 probability.",
        example=1,
        json_schema_extra={
            "example": 1
        }
    )
    risk_category: Optional[RiskCategory] = Field(
        None,
        description="Risk categorization: "
        "LOW (0-0.33), MEDIUM (0.33-0.66), HIGH (0.66-1.0)",
        example="HIGH",
        json_schema_extra={
            "example": "HIGH"
        }
    )
    risk_score: Optional[int] = Field(
        None,
        ge=0,
        le=100,
        description="Risk score from 0-100 (100 being highest risk). "
        "Calculated as probability × 100.",
        example=67,
        json_schema_extra={
            "example": 67
        }
    )
    recommendation: Optional[str] = Field(
        None,
        description="Business recommendation based on risk assessment.",
        example="Reject - High risk of default",
        json_schema_extra={
            "example": "Reject - High risk of default"
        }
    )
    customer_id: Optional[str] = Field(
        None,
        description="Echoed customer ID if provided in request.",
        example="CUSTOMER_001",
        json_schema_extra={
            "example": "CUST_001"
        }
    )
    model: Optional[str] = Field(
        None,
        description="Model identifier and source used for prediction.",
        example="mlflow:models:/credit_risk_model/Production",
        json_schema_extra={
            "example": "local:models/logistic_champion_fixed.joblib"
        }
    )
    timestamp: Optional[str] = Field(
        None,
        description="ISO 8601 timestamp of when prediction was made.",
        example="2024-01-15T10:30:45.123Z",
        json_schema_extra={
            "example": "2024-01-15T10:30:45.123Z"
        }
    )
    features_used: Optional[list] = Field(
        None,
        description="List of features used by the model for this prediction.",
        example=["Year_mean", "Month_mean"],
        json_schema_extra={
            "example": ["Year_mean", "Month_mean"]
        }
    )

    class Config:
        schema_extra = {
            "example": {
                "probability": 0.6669,
                "predicted_class": 1,
                "risk_category": "HIGH",
                "risk_score": 67,
                "recommendation": "Reject - High risk of default",
                "customer_id": "CUST_001",
                "model": "mlflow:models:/credit_risk_model/Production",
                "timestamp": "2024-01-15T10:30:45.123Z",
                "features_used": ["Year_mean", "Month_mean"]
            }
        }


class HealthResponse(BaseModel):
    """ Health check response model. """
    status: str = Field(..., description="Overall system status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_source: Optional[str] = Field(
        None, description="Source of the loaded model")
    model_type: Optional[str] = Field(
        None, description="Type of model (MLflow or Local)")
    feature_count: Optional[int] = Field(
        None, description="Number of features expected by model")
    timestamp: str = Field(
        ..., description="ISO 8601 timestamp of health check")


class ModelInfoResponse(BaseModel):
    """ Model information response model. """
    model_source: str = Field(..., description="Source of the loaded model")
    model_type: Optional[str] = Field(
        None, description="Type of model")
    features_required: Optional[list] = Field(
        None, description="Features required by the model")
    feature_count: Optional[int] = Field(
        None, description="Number of features expected")
    risk_categories: Dict[str, str] = Field(
        ..., description="Risk category thresholds")
    risk_recommendations: Dict[str, str] = Field(
        ..., description="Business recommendations per risk category")
