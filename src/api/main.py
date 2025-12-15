"""
FastAPI app for serving the trained risk model.

Behavior:
- On startup, tries to load model from MLflow Model Registry using MODEL_NAME and MODEL_STAGE env vars.
- If unavailable, falls back to loading a local joblib pickle/model at MODEL_LOCAL_PATH.
- Exposes POST /predict that accepts PredictionRequest (features dict) and returns PredictionResponse.
"""
import os
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.requests import Request

# local pydantic models
from api.pydantic_models import PredictionRequest, PredictionResponse

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="PTV Risk API", version="0.1.0")

# Configuration via environment variables
MODEL_NAME = os.getenv("MODEL_NAME", "")
MODEL_STAGE = os.getenv("MODEL_STAGE", "production")
MODEL_LOCAL_PATH = os.getenv("MODEL_LOCAL_PATH", "")  # fallback
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", None)

# Runtime loaded model (global)
MODEL = None
MODEL_FEATURES: Optional[List[str]] = None
MODEL_SOURCE: str = "none"


def load_model_from_mlflow(name: str, stage: str):
    import mlflow
    from mlflow.pyfunc import load_model as mlflow_load_model

    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{name}/{stage}"
    logger.info("Loading model from MLflow URI %s", model_uri)
    return mlflow_load_model(model_uri)


def load_model_local(path: str):
    import joblib

    logger.info("Loading local model from %s", path)
    return joblib.load(path)


@app.on_event("startup")
def load_model():
    global MODEL, MODEL_SOURCE, MODEL_FEATURES
    # Try MLflow model registry if configured
    try:
        if MODEL_NAME:
            logger.info(
                "Attempting to load model '%s' (stage=%s) from MLflow...", MODEL_NAME, MODEL_STAGE)
            MODEL = load_model_from_mlflow(MODEL_NAME, MODEL_STAGE)
            MODEL_SOURCE = f"mlflow:{MODEL_NAME}/{MODEL_STAGE}"
    except Exception as exc:
        logger.warning("Failed to load model from MLflow: %s", exc)
        MODEL = None

    # If model still None, try local path
    if MODEL is None and MODEL_LOCAL_PATH:
        try:
            MODEL = load_model_local(MODEL_LOCAL_PATH)
            MODEL_SOURCE = f"local:{MODEL_LOCAL_PATH}"
        except Exception as exc:
            logger.error("Failed to load local model: %s", exc)
            MODEL = None

    if MODEL is None:
        logger.error(
            "No model loaded. Predictions will fail until a model is provided.")
    else:
        # Try to infer expected feature columns from model metadata if available
        try:
            # mlflow pyfunc model may expose metadata via _model_impl
            if hasattr(MODEL, "metadata") and hasattr(MODEL.metadata, "signature") and MODEL.metadata.signature:
                signature = MODEL.metadata.signature
                if signature.inputs:
                    MODEL_FEATURES = [c.name for c in signature.inputs]
            # if sklearn Pipeline, try to extract from a stored attribute 'feature_names_in_'
            elif hasattr(MODEL, "feature_names_in_"):
                MODEL_FEATURES = list(MODEL.feature_names_in_)
            # else keep None
        except Exception:
            MODEL_FEATURES = None
        logger.info("Model loaded from %s. Expected features: %s",
                    MODEL_SOURCE, MODEL_FEATURES)


def _build_dataframe_from_features(features: Dict[str, Any]) -> pd.DataFrame:
    # Convert to single-row DataFrame; ensure consistent column ordering if MODEL_FEATURES known
    row = pd.DataFrame([features])
    if MODEL_FEATURES is not None:
        missing = [c for c in MODEL_FEATURES if c not in row.columns]
        if missing:
            raise ValueError(f"Missing required feature(s): {missing}")
        # reorder columns to model expectation
        row = row[MODEL_FEATURES]
    return row


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        X = _build_dataframe_from_features(payload.features)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        # Prefer predict_proba when available
        proba = None
        pred_class = None

        # MLflow pyfunc models respond to .predict
        if hasattr(MODEL, "predict_proba"):
            proba = MODEL.predict_proba(X)[:, 1]
            pred_class = MODEL.predict(X)
        else:
            # Try predict_proba via underlying model if wrapped
            try:
                proba = MODEL.predict_proba(X)[:, 1]
            except Exception:
                # try pyfunc predict (may return probabilities or labels depending on model)
                res = MODEL.predict(X)
                # If result is array-like of shape (n, 2) or (n, ) handle accordingly
                if hasattr(res, "shape") and getattr(res, "ndim", 1) == 2 and res.shape[1] == 2:
                    # assume probabilities returned
                    proba = res[:, 1]
                    pred_class = (proba >= 0.5).astype(int)
                else:
                    # fallback: we have predicted label only
                    pred_class = res
                    proba = None

        prob_value = float(proba[0]) if proba is not None else None
        pred_value = int(pred_class[0]) if pred_class is not None else None

        resp = PredictionResponse(
            probability=prob_value if prob_value is not None else 0.0,
            predicted_class=pred_value,
            customer_id=payload.customer_id,
            model=MODEL_SOURCE
        )
        return resp
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")
