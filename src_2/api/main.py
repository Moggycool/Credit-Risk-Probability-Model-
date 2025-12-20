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

from src_2.api.pydantic_models import PredictionRequest, PredictionResponse
from src_2.api.predictor import get_predictor, ModelNotLoadedError

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the model at startup.
    Fail fast if the model cannot be loaded.
    Supports both local models and MLflow Model Registry.
    """
    try:
        predictor = get_predictor()

        # Check what type of model was loaded
        model_type = "MLflow Model Registry" if "mlflow:" in predictor.source else "local file"

        logger.info(
            "Startup successful | model_source=%s | model_type=%s | feature_count=%s",
            predictor.source,
            model_type,
            len(predictor.features) if predictor.features else "unknown",
        )

        if predictor.features:
            logger.info("Model features: %s", predictor.features)

    except ImportError as exc:
        if "mlflow" in str(exc).lower():
            logger.error(
                "MLflow not installed. For MLflow Model Registry support, "
                "install with: pip install mlflow"
            )
            logger.error("Falling back to local model if available")
            # Try to continue with local model if environment variable is set
            import os
            if os.getenv("MODEL_LOCAL_PATH"):
                logger.info("Attempting to load local model as fallback")
                # Re-raise to trigger container restart or handle appropriately
                raise
        raise

    except Exception as exc:
        logger.exception("Startup failed: model could not be loaded: %s", exc)
        raise
    yield


app = FastAPI(
    title="Credit Risk Scoring API",
    version="1.0.0",
    description="API for credit risk prediction. "
                "Supports both local models and MLflow Model Registry.",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """Health check endpoint."""
    try:
        predictor = get_predictor()
        return JSONResponse(
            content={
                "status": "ok",
                "model_loaded": predictor.model is not None,
                "model_source": predictor.source,
                "model_type": "MLflow" if "mlflow:" in predictor.source else "Local",
                "feature_count": len(predictor.features)
                if predictor.features
                else None,
            }
        )
    except Exception:
        return JSONResponse(
            content={"status": "ok", "model_loaded": False},
            status_code=200,
        )


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    """
    Perform credit risk prediction.

    Perform credit risk prediction.

    Supports both:
    1. Local model file (MODEL_LOCAL_PATH)
    2. MLflow Model Registry (MODEL_MLFLOW_URI)

    Required features for current model:
    - Year_mean: Average transaction year
    - Month_mean: Average transaction month

    """
    try:
        predictor = get_predictor()
        logger.info("Prediction request for customer: %s", payload.customer_id)

        result = predictor.predict(payload.features)

        return PredictionResponse(
            probability=result["probability"],
            predicted_class=result["predicted_class"],
            risk_category=result["risk_category"],
            risk_score=result["risk_score"],
            recommendation=result["recommendation"],
            customer_id=payload.customer_id,
            model=predictor.source,
            timestamp=datetime.now().isoformat(),
            features_used=predictor.features,
        )

    except ModelNotLoadedError:
        raise HTTPException(status_code=503, detail="Model not loaded")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction failed")
