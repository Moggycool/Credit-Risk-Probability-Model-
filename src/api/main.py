"""
FastAPI app for serving the trained risk model (lifespan-based startup).

Behavior:
- Uses the Predictor utility (api.predictor) to load the model (MLflow or local joblib) at app startup via lifespan.
- Exposes GET /health and POST /predict endpoints.
"""
import logging
from typing import Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from api.pydantic_models import PredictionRequest, PredictionResponse
from api.predictor import get_predictor, ModelNotLoadedError

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan handler that attempts to preload the predictor on startup.
    This replaces the deprecated @app.on_event("startup").
    """
    try:
        predictor = get_predictor()
        logger.info("Predictor loaded from source: %s", predictor.source)
    except ModelNotLoadedError as exc:
        logger.error("Model not loaded at startup: %s", exc)
    except Exception as exc:
        logger.exception("Unexpected error while loading predictor at startup: %s", exc)
    yield
    # Optionally add shutdown cleanup here


app = FastAPI(title="PTV Risk API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    """Simple health check endpoint."""
    try:
        predictor = get_predictor()
        ready = predictor.model is not None
    except Exception:
        ready = False
    return JSONResponse(content={"status": "ok", "model_loaded": ready})


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    """
    Accept a PredictionRequest with a 'features' dict (feature_name -> numeric value).
    Returns PredictionResponse with probability, predicted_class, customer_id, and model source.
    """
    try:
        predictor = get_predictor()
    except ModelNotLoadedError:
        raise HTTPException(status_code=503, detail="Model not loaded")
    except Exception as exc:
        logger.exception("Failed to obtain predictor: %s", exc)
        raise HTTPException(
            status_code=500, detail="Internal error obtaining predictor")

    try:
        result = predictor.predict(payload.features)
    except ValueError as exc:
        # typically missing features or mismatched schema
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    resp = PredictionResponse(
        probability=result.get("probability"),
        predicted_class=result.get("predicted_class"),
        customer_id=payload.customer_id,
        model=predictor.source,
    )
    return resp
