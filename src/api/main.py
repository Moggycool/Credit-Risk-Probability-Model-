"""
FastAPI application for Credit Risk Scoring.

- Loads the predictor at startup (fail-fast).
- Exposes health and prediction endpoints.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.api.pydantic_models import PredictionRequest, PredictionResponse
from src.api.predictor import get_predictor, ModelNotLoadedError

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the model at startup.
    Fail fast if the model cannot be loaded.
    """
    try:
        predictor = get_predictor()
        logger.info(
            "Startup successful | model_source=%s | feature_count=%s",
            predictor.source,
            len(predictor.features) if predictor.features else "unknown",
        )
        if predictor.features:
            logger.info("Model features: %s", predictor.features)
    except Exception as exc:
        logger.exception("Startup failed: model could not be loaded: %s", exc)
        raise
    yield


app = FastAPI(
    title="Credit Risk Scoring API",
    version="1.0.0",
    description="API for credit risk prediction using engineered features. Predicts probability of default and provides risk categories.",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    """Root endpoint with API information."""
    try:
        predictor = get_predictor()
        return {
            "message": "Credit Risk Scoring API",
            "version": "1.0.0",
            "status": "operational",
            "model_loaded": predictor.model is not None,
            "model_source": predictor.source,
            "features_required": predictor.features if predictor.features else "unknown",
            "endpoints": {
                "GET /": "API information",
                "GET /health": "Health check with model status",
                "POST /predict": "Make credit risk prediction",
                "GET /model-info": "Get model information",
                "GET /docs": "API documentation (Swagger UI)",
                "GET /redoc": "Alternative API documentation",
            }
        }
    except Exception:
        return {
            "message": "Credit Risk Scoring API",
            "version": "1.0.0",
            "status": "starting_up",
            "model_loaded": False,
        }


@app.get("/health")
def health():
    """Health check endpoint with detailed model information."""
    try:
        predictor = get_predictor()
        return JSONResponse(
            content={
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "model_loaded": predictor.model is not None,
                "model_source": predictor.source,
                "feature_count": len(predictor.features) if predictor.features else None,
                "features_required": predictor.features if predictor.features else None,
                "risk_thresholds": predictor.risk_thresholds if hasattr(predictor, 'risk_thresholds') else None,
            }
        )
    except Exception as exc:
        logger.error("Health check failed: %s", exc)
        return JSONResponse(
            content={
                "status": "degraded",
                "timestamp": datetime.now().isoformat(),
                "model_loaded": False,
                "error": str(exc)
            },
            status_code=503,
        )


@app.get("/model-info")
def model_info():
    """Get detailed information about the loaded model."""
    try:
        predictor = get_predictor()

        info = {
            "model_source": predictor.source,
            "features_required": predictor.features if predictor.features else None,
            "feature_count": predictor.n_features_expected,
            "risk_categories": {
                "LOW": f"0.0 - {predictor.risk_thresholds['low_max']}",
                "MEDIUM": f"{predictor.risk_thresholds['low_max']} - {predictor.risk_thresholds['medium_max']}",
                "HIGH": f"{predictor.risk_thresholds['medium_max']} - {predictor.risk_thresholds['high_max']}",
            },
            "risk_recommendations": {
                "LOW": "Approve - Low risk of default",
                "MEDIUM": "Review - Moderate risk, consider additional verification",
                "HIGH": "Reject - High risk of default"
            }
        }

        # Add model-specific information if available
        if predictor.model is not None:
            model = predictor.model
            info["model_type"] = type(model).__name__

            if hasattr(model, 'n_features_in_'):
                info["model_n_features"] = model.n_features_in_

            if hasattr(model, 'classes_'):
                info["model_classes"] = model.classes_.tolist()

        return info

    except Exception as exc:
        logger.error("Model info failed: %s", exc)
        raise HTTPException(
            status_code=503, detail=f"Model information unavailable: {exc}")


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    """
    Perform credit risk prediction.

    Required features:
    - Year_mean: Average transaction year (e.g., 2019.0)
    - Month_mean: Average transaction month (e.g., 8.0)

    Returns:
    - probability: Risk probability (0.0 to 1.0)
    - predicted_class: Binary classification (0=Low Risk, 1=High Risk)
    - risk_category: Risk category (LOW, MEDIUM, HIGH)
    - risk_score: Risk score from 0-100
    - recommendation: Business recommendation
    """
    try:
        predictor = get_predictor()
        logger.info("Prediction request for customer: %s", payload.customer_id)
        logger.debug("Received features: %s", payload.features)

        # Make prediction
        result = predictor.predict(payload.features)

        # Log prediction result
        logger.info(
            "Prediction completed | customer=%s | probability=%.4f | category=%s | score=%d",
            payload.customer_id,
            result["probability"] if result["probability"] is not None else 0,
            result["risk_category"],
            result["risk_score"]
        )

        # Prepare response
        response_data = {
            "probability": result["probability"],
            "predicted_class": result["predicted_class"],
            "risk_category": result["risk_category"],
            "risk_score": result["risk_score"],
            "recommendation": result["recommendation"],
            "customer_id": payload.customer_id,
            "model": result["model_source"],
            "timestamp": datetime.now().isoformat(),
        }

        return PredictionResponse(**response_data)

    except ModelNotLoadedError:
        logger.error("Model not loaded for prediction request")
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please try again later.")

    except ValueError as exc:
        logger.error("Invalid input for prediction: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}"
        )


@app.get("/predict/batch-example")
def batch_example():
    """
    Example of different risk scenarios.
    Useful for testing and understanding the model behavior.
    """
    examples = [
        {
            "description": "Low Risk - Early 2018",
            "customer_id": "EXAMPLE_LOW_001",
            "features": {"Year_mean": 2018.0, "Month_mean": 3.0}
        },
        {
            "description": "Medium Risk - Mid 2018",
            "customer_id": "EXAMPLE_MEDIUM_001",
            "features": {"Year_mean": 2018.5, "Month_mean": 6.5}
        },
        {
            "description": "Medium-High Risk - Early 2019",
            "customer_id": "EXAMPLE_MEDHIGH_001",
            "features": {"Year_mean": 2019.0, "Month_mean": 6.0}
        },
        {
            "description": "High Risk - Late 2019",
            "customer_id": "EXAMPLE_HIGH_001",
            "features": {"Year_mean": 2019.0, "Month_mean": 12.0}
        }
    ]

    return {
        "message": "Example prediction scenarios",
        "note": "These are examples only. Use POST /predict for actual predictions.",
        "examples": examples,
        "feature_notes": {
            "Year_mean": "Higher values (more recent years) indicate higher risk",
            "Month_mean": "Higher values (later in year) indicate higher risk"
        }
    }
