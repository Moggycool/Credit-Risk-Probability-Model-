"""
Model loader and predictor utilities for the PTV API.

Responsibilities:
- Load a model from MLflow Model Registry (models:/<name>/<stage>) or local joblib fallback.
- Expose a simple Predictor class with .load() and .predict(features_dict).
- Provide get_predictor() that FastAPI can call to obtain a ready-to-use predictor.

Behavior changes in this updated version:
- If the model exposes an expected feature list (via signature or feature_names_in_),
  the predictor will accept partial feature dicts from clients: missing features are
  filled with NaN and the model/pipeline's imputer(s) can handle them.
- If the model does not expose feature names, the predictor will build a DataFrame
  from the provided features (so callers must provide exactly what the model expects).
- The predictor will also attempt to load a saved feature_names JSON file located at
  MODEL_FEATURES_PATH (default: ./models/feature_names.json) if available.
"""
from typing import Any, Dict, List, Optional
import os
import logging
import joblib
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ModelNotLoadedError(RuntimeError):
    pass


class Predictor:
    def __init__(self) -> None:
        self.model = None
        self.features: Optional[List[str]] = None
        self.source: str = "none"

    def _load_mlflow_model(self, model_name: str, model_stage: str, tracking_uri: Optional[str]):
        """Attempt to load model from MLflow registry; returns loaded model or raises."""
        import mlflow
        from mlflow.pyfunc import load_model as mlflow_load_model

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        model_uri = f"models:/{model_name}/{model_stage}"
        logger.info("Loading model from MLflow URI=%s", model_uri)
        return mlflow_load_model(model_uri)

    def _load_local_model(self, path: str):
        """Load a local joblib model from disk."""
        logger.info("Loading local model from %s", path)
        return joblib.load(path)

    def _load_feature_names_from_file(self, path: str) -> Optional[List[str]]:
        try:
            if path and os.path.exists(path):
                import json

                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    logger.info("Loaded feature names from %s", path)
                    return list(data)
        except Exception as exc:
            logger.warning("Failed to load feature names from %s: %s", path, exc)
        return None

    def load(self) -> None:
        """
        Try to load the model from MLflow registry first, then local joblib path.
        Environment variables are read at call-time to allow tests or runtime to set them.
        """
        MODEL_NAME = os.getenv("MODEL_NAME", "").strip()
        MODEL_STAGE = os.getenv("MODEL_STAGE", "production").strip()
        MODEL_LOCAL_PATH = os.getenv("MODEL_LOCAL_PATH", "").strip()
        MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "").strip()
        MODEL_FEATURES_PATH = os.getenv(
            "MODEL_FEATURES_PATH", "models/feature_names.json").strip()

        # Optional mlflow availability check
        try:
            import mlflow  # noqa: F401
            mlflow_available = True
        except Exception:
            mlflow_available = False

        # Try MLflow registry if a model name is provided and mlflow available
        if MODEL_NAME:
            if not mlflow_available:
                logger.warning("mlflow not available; skipping MLflow model load")
            else:
                try:
                    self.model = self._load_mlflow_model(
                        MODEL_NAME, MODEL_STAGE, MLFLOW_TRACKING_URI or None)
                    self.source = f"mlflow:{MODEL_NAME}/{MODEL_STAGE}"
                except Exception as exc:
                    logger.warning("Could not load model from MLflow: %s", exc)
                    self.model = None

        # Fallback to local joblib if MLflow load failed or not attempted
        if self.model is None and MODEL_LOCAL_PATH:
            try:
                self.model = self._load_local_model(MODEL_LOCAL_PATH)
                self.source = f"local:{MODEL_LOCAL_PATH}"
            except Exception as exc:
                logger.error("Failed loading local model: %s", exc)
                self.model = None

        if self.model is None:
            raise ModelNotLoadedError(
                "No model loaded; set MODEL_NAME or MODEL_LOCAL_PATH")

        # Infer expected features if model exposes them or via provided JSON
        self.features = None
        try:
            # MLflow pyfunc may include metadata.signature.inputs
            if hasattr(self.model, "metadata") and getattr(self.model.metadata, "signature", None):
                sig = self.model.metadata.signature
                if getattr(sig, "inputs", None):
                    self.features = [c.name for c in sig.inputs]
            # sklearn pipelines often expose feature_names_in_
            if self.features is None and hasattr(self.model, "feature_names_in_"):
                try:
                    self.features = list(getattr(self.model, "feature_names_in_"))
                except Exception:
                    self.features = None
            # fallback: try to load a saved JSON file with feature names
            if self.features is None:
                file_feats = self._load_feature_names_from_file(MODEL_FEATURES_PATH)
                if file_feats:
                    self.features = file_feats
        except Exception as exc:
            logger.debug("Failed to infer feature names: %s", exc)
            self.features = None

        logger.info("Model loaded from %s; expected features=%s",
                    self.source, self.features)

    def _build_df(self, features: Dict[str, Any]) -> pd.DataFrame:
        """
        Return one-row DataFrame:
        - If self.features is known: build a row with those columns in that order, filling missing with np.nan.
          Extra keys present in `features` but not in self.features will be ignored (and logged).
        - If self.features is unknown: build DataFrame directly from provided features (caller must provide correct keys).
        """
        if self.features is None:
            # no known expected feature set, trust caller (DataFrame will have caller-provided order)
            row = pd.DataFrame([features])
            return row

        # Build ordered dict of expected features, fill missing values with NaN
        row_dict = {}
        extra_keys = []
        for f in self.features:
            if f in features:
                row_dict[f] = features[f]
            else:
                row_dict[f] = np.nan
        # detect extras to help users debug
        for k in features.keys():
            if k not in self.features:
                extra_keys.append(k)
        if extra_keys:
            logger.debug(
                "Input contained extra features not expected by model; ignoring: %s", extra_keys)

        row = pd.DataFrame([row_dict], columns=self.features)
        return row

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict probability and class for given features dict.
        Returns: {"probability": float or None, "predicted_class": int or None}
        """
        if self.model is None:
            raise ModelNotLoadedError("Model not loaded. Call load() first.")

        X = self._build_df(features)

        proba = None
        pred = None

        try:
            # If model exposes predict_proba, prefer that
            if hasattr(self.model, "predict_proba"):
                proba_arr = self.model.predict_proba(X)
                # predict_proba may return shape (n,2) or (n,). handle both
                if getattr(proba_arr, "ndim", 1) == 2 and proba_arr.shape[1] >= 2:
                    proba = proba_arr[:, 1]
                else:
                    # sometimes predict_proba returns single-col probabilities
                    proba = proba_arr.ravel()
                pred = self.model.predict(X)
            else:
                # Some pyfunc models return probabilities directly from predict()
                res = self.model.predict(X)
                # If res is ndarray with two columns, treat second as probability
                if hasattr(res, "ndim") and getattr(res, "ndim", 1) == 2 and res.shape[1] >= 2:
                    proba = res[:, 1]
                    pred = (proba >= 0.5).astype(int)
                else:
                    # treat res as class predictions (or scalar)
                    pred = res
                    proba = None
        except Exception as exc:
            logger.exception("Prediction failed: %s", exc)
            # Reraise to allow FastAPI to send 500
            raise

        # Normalize outputs to python scalars
        prob_val = float(proba[0]) if proba is not None else None
        # pred may be array-like or scalar; handle safely
        try:
            if hasattr(pred, "__len__"):
                pred_val = int(pred[0])
            else:
                pred_val = int(pred)
        except Exception:
            pred_val = None

        return {"probability": prob_val, "predicted_class": pred_val}


# Global singleton used by FastAPI
_predictor = Predictor()


def get_predictor() -> Predictor:
    """Return the global predictor, loading the model on demand."""
    global _predictor
    if _predictor.model is None:
        _predictor.load()
    return _predictor
