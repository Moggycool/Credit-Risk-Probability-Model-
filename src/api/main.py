"""
FastAPI app for serving the trained risk model (lifespan-based startup).

Behavior:
- Uses the Predictor utility (api.predictor) to load the model (MLflow or local joblib) at app startup via lifespan.
- Exposes GET /health and POST /predict endpoints.
- Fail-fast on startup if the model cannot be loaded (so container startup fails early).
"""
import logging
from typing import Any
from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

try:
    # Prefer relative import when running as a package
    from .pydantic_models import PredictionRequest, PredictionResponse
except ImportError:
    # Fallback to absolute import when running the module as a script
    try:
        from api.pydantic_models import PredictionRequest, PredictionResponse
    except ImportError:
        from pydantic_models import PredictionRequest, PredictionResponse

try:
    # Prefer relative import when running as a package
    from .predictor import get_predictor, ModelNotLoadedError
except ImportError:
    # Fallback to absolute import when running the module as a script
    try:
        from api.predictor import get_predictor, ModelNotLoadedError
    except ImportError:
        from predictor import get_predictor, ModelNotLoadedError

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan handler that attempts to preload the predictor on startup.
    This replaces the deprecated @app.on_event("startup").

    Behavior:
     - Attempts to load the model via get_predictor() so model loading happens
       during container startup.
     - Logs model source and expected feature names.
     - Raises if model not loaded (fail-fast) so orchestration will mark the
       service as failed rather than run an unhealthy instance.
    """
    try:
        predictor = get_predictor()
        logger.info(
            "Predictor loaded from source: %s; features: %s",
            predictor.source,
            predictor.features,
        )
        # If you want to enforce that a feature list is present, uncomment:
        # if predictor.features is None:
        #     raise ModelNotLoadedError("Model loaded but expected feature list unknown")
    except ModelNotLoadedError as exc:
        # Fail-fast: re-raise so the app startup fails and container exits
        logger.exception("Model not loaded at startup (failing fast): %s", exc)
        raise
    except Exception as exc:
        logger.exception("Unexpected error while loading predictor at startup: %s", exc)
        # Fail-fast for unexpected errors as well
        raise
    yield
    # Optionally add shutdown cleanup here


app = FastAPI(title="PTV Risk API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    """Simple health check endpoint. Returns whether the model is loaded."""
    try:
        predictor = get_predictor()
        ready = predictor.model is not None
        source = predictor.source
        features = predictor.features
    except Exception:
        ready = False
        source = None
        features = None
    return JSONResponse(content={"status": "ok", "model_loaded": ready, "model_source": source, "feature_count": len(features) if features else None})


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    """
    Accept a PredictionRequest with a 'features' dict (feature_name -> numeric value).
    Returns PredictionResponse with probability, predicted_class, customer_id, and model source.
    """
    try:
        predictor = get_predictor()
    except ModelNotLoadedError:
        # model wasn't loaded at startup or cannot be loaded on demand
        raise HTTPException(status_code=503, detail="Model not loaded")
    except Exception as exc:
        logger.exception("Failed to obtain predictor: %s", exc)
        raise HTTPException(
            status_code=500, detail="Internal error obtaining predictor")

    try:
        result = predictor.predict(payload.features)
    except ValueError as exc:
        # typically missing features or mismatched schema
        # include model expected features for easier debugging (if available)
        expected = getattr(get_predictor(), "features", None)
        detail_msg = str(exc)
        if expected:
            detail_msg = f"{detail_msg}; expected_features_count={len(expected)}"
        raise HTTPException(status_code=400, detail=detail_msg)
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
