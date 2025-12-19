"""
Model loader and predictor utilities for the Credit Risk API.

Responsibilities:
- Load a model for inference (local joblib by default; optional MLflow Registry).
- Infer expected feature names when possible.
- Accept partial feature dictionaries and align them safely to the model schema.
- Expose a global, lazily-loaded Predictor instance for FastAPI.

Design notes:
- Local joblib loading is the default and recommended production path.
- MLflow Registry loading is supported only when explicitly configured.
- Missing features are filled with NaN and handled by preprocessing pipelines.
"""

from typing import Any, Dict, List, Optional
import os
import logging
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ModelNotLoadedError(RuntimeError):
    """Raised when no model can be loaded for inference."""
    pass


class Predictor:
    def __init__(self) -> None:
        self.model = None
        self.features: Optional[List[str]] = None
        self.source: str = "none"

    # ------------------------------------------------------------------
    # Model loading helpers
    # ------------------------------------------------------------------
    def _load_local_model(self, path: str):
        logger.info("Loading local model from %s", path)
        return joblib.load(path)

    def _load_mlflow_model(
        self,
        model_name: str,
        model_stage: str,
        tracking_uri: Optional[str],
    ):
        import mlflow
        from mlflow.pyfunc import load_model as mlflow_load_model

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        model_uri = f"models:/{model_name}/{model_stage}"
        logger.info("Loading MLflow model from %s", model_uri)
        return mlflow_load_model(model_uri)

    def _load_feature_names_from_file(self, path: str) -> Optional[List[str]]:
        try:
            if path and os.path.exists(path):
                import json

                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    logger.info("Loaded feature names from %s", path)
                    return list(data)
        except Exception as exc:
            logger.warning("Failed to load feature names from %s: %s", path, exc)
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load(self) -> None:
        """
        Load the model for inference.

        Priority:
        1. Local joblib model (MODEL_LOCAL_PATH)  ← recommended
        2. MLflow Registry model (MODEL_NAME + MODEL_STAGE)
        """
        MODEL_LOCAL_PATH = os.getenv("MODEL_LOCAL_PATH", "").strip()
        MODEL_NAME = os.getenv("MODEL_NAME", "").strip()
        MODEL_STAGE = os.getenv("MODEL_STAGE", "production").strip()
        MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "").strip()
        MODEL_FEATURES_PATH = os.getenv(
            "MODEL_FEATURES_PATH", "models/feature_names.json"
        ).strip()

        # ---------------------------
        # 1️⃣ Local model (default)
        # ---------------------------
        if MODEL_LOCAL_PATH:
            try:
                self.model = self._load_local_model(MODEL_LOCAL_PATH)
                self.source = f"local:{MODEL_LOCAL_PATH}"
            except Exception as exc:
                logger.error("Failed to load local model: %s", exc)
                self.model = None

        # ---------------------------
        # 2️⃣ MLflow Registry (opt-in)
        # ---------------------------
        if self.model is None and MODEL_NAME:
            try:
                self.model = self._load_mlflow_model(
                    MODEL_NAME, MODEL_STAGE, MLFLOW_TRACKING_URI or None
                )
                self.source = f"mlflow:{MODEL_NAME}/{MODEL_STAGE}"
            except Exception as exc:
                logger.error("Failed to load MLflow model: %s", exc)
                self.model = None

        if self.model is None:
            raise ModelNotLoadedError(
                "No model loaded. Set MODEL_LOCAL_PATH (recommended)."
            )

        # ---------------------------
        # Infer expected features
        # ---------------------------
        self.features = None
        try:
            # sklearn pipelines often expose this
            if hasattr(self.model, "feature_names_in_"):
                self.features = list(self.model.feature_names_in_)

            # fallback: explicit feature file
            if self.features is None:
                self.features = self._load_feature_names_from_file(
                    MODEL_FEATURES_PATH
                )
        except Exception:
            self.features = None

        logger.info(
            "Model ready | source=%s | feature_count=%s",
            self.source,
            len(self.features) if self.features else "unknown",
        )

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------
    def _build_dataframe(self, features: Dict[str, Any]) -> pd.DataFrame:
        """
        Build a single-row DataFrame aligned to expected features.

        - If expected features are known:
            * Missing → NaN
            * Extra → ignored
        - If unknown:
            * Trust caller input
        """
        if self.features is None:
            return pd.DataFrame([features])

        row = {
            feature: features.get(feature, np.nan)
            for feature in self.features
        }

        return pd.DataFrame([row], columns=self.features)

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference.

        Returns:
            {
              "probability": float | None,
              "predicted_class": int | None
            }
        """
        if self.model is None:
            raise ModelNotLoadedError("Model not loaded")

        X = self._build_dataframe(features)

        try:
            if hasattr(self.model, "predict_proba"):
                proba_arr = self.model.predict_proba(X)
                probability = (
                    float(proba_arr[0, 1])
                    if proba_arr.ndim == 2
                    else float(proba_arr[0])
                )
                predicted_class = int(probability >= 0.5)
            else:
                preds = self.model.predict(X)
                predicted_class = int(preds[0])
                probability = None
        except Exception as exc:
            logger.exception("Prediction failed: %s", exc)
            raise

        return {
            "probability": probability,
            "predicted_class": predicted_class,
        }


# ----------------------------------------------------------------------
# Global singleton used by FastAPI
# ----------------------------------------------------------------------
_predictor = Predictor()


def get_predictor() -> Predictor:
    """Return a loaded Predictor instance (lazy load)."""
    if _predictor.model is None:
        _predictor.load()
    return _predictor
