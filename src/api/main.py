"""
FastAPI application for Credit Risk Scoring.

- Loads the predictor at startup (fail-fast).
- Exposes health and prediction endpoints.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

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
    except Exception as exc:
        logger.exception("Startup failed: model could not be loaded: %s", exc)
        raise
    yield


app = FastAPI(
    title="Credit Risk Scoring API",
    version="1.0.0",
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
    """
    try:
        predictor = get_predictor()
        result = predictor.predict(payload.features)
    except ModelNotLoadedError:
        raise HTTPException(status_code=503, detail="Model not loaded")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction failed")

    return PredictionResponse(
        probability=result["probability"],
        predicted_class=result["predicted_class"],
        customer_id=payload.customer_id,
        model=predictor.source,
    )
