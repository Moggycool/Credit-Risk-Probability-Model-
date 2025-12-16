"""
Model loader and predictor utilities for the PTV API.

Responsibilities:
- Load a model from MLflow Model Registry (models:/<name>/<stage>) or local joblib fallback.
- Expose a simple Predictor class with .load() and .predict(features_dict).
- Provide get_predictor() that FastAPI can call to obtain a ready-to-use predictor.
"""
from typing import Any, Dict, List, Optional
import os
import logging
import joblib
import pandas as pd

logger = logging.getLogger(__name__)


class ModelNotLoadedError(RuntimeError):
    pass


class Predictor:
    def __init__(self):
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

    def load(self) -> None:
        """
        Try to load the model from MLflow registry first, then local joblib path.
        Environment variables are read at call-time to allow tests or runtime to set them.
        """
        # Read env vars at runtime (avoids import-time capture issues)
        MODEL_NAME = os.getenv("MODEL_NAME", "").strip()
        MODEL_STAGE = os.getenv("MODEL_STAGE", "production").strip()
        MODEL_LOCAL_PATH = os.getenv("MODEL_LOCAL_PATH", "").strip()
        MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "").strip()

        # Optional mlflow availability check
        try:
            import mlflow  # noqa: F401
            mlflow_available = True
        except Exception:
            mlflow_available = False

        # Try MLflow registry if a model name is provided and mlflow available
        if MODEL_NAME:
            if not mlflow_available:
                logger.warning(
                    "mlflow not available; skipping MLflow model load")
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

        # Infer expected features if model exposes them
        try:
            # MLflow pyfunc with signature
            if hasattr(self.model, "metadata") and getattr(self.model.metadata, "signature", None):
                sig = self.model.metadata.signature
                if sig.inputs:
                    self.features = [c.name for c in sig.inputs]
            # sklearn models/pipelines often have feature_names_in_
            elif hasattr(self.model, "feature_names_in_"):
                self.features = list(self.model.feature_names_in_)
            else:
                self.features = None
            logger.info("Model loaded from %s; expected features=%s",
                        self.source, self.features)
        except Exception:
            self.features = None

    def _build_df(self, features: Dict[str, Any]) -> pd.DataFrame:
        """Return one-row DataFrame ordered per self.features if available."""
        row = pd.DataFrame([features])
        if self.features is not None:
            missing = [f for f in self.features if f not in row.columns]
            if missing:
                raise ValueError(f"Missing required feature(s): {missing}")
            row = row[self.features]
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

        # Prefer predict_proba where available
        try:
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(X)[:, 1]
                pred = self.model.predict(X)
            else:
                res = self.model.predict(X)
                # If predict returns 2-col array, treat as probabilities
                if hasattr(res, "shape") and getattr(res, "ndim", 1) == 2 and res.shape[1] == 2:
                    proba = res[:, 1]
                    pred = (proba >= 0.5).astype(int)
                else:
                    pred = res
                    proba = None
        except Exception:
            logger.exception("Prediction failed")
            raise

        return {
            "probability": float(proba[0]) if proba is not None else None,
            "predicted_class": int(pred[0]) if hasattr(pred, "__len__") else int(pred)
        }


# Global singleton used by FastAPI
_predictor = Predictor()


def get_predictor() -> Predictor:
    """Return the global predictor, loading the model on demand."""
    global _predictor
    if _predictor.model is None:
        _predictor.load()
    return _predictor
