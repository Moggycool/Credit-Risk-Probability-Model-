"""
Model loader and predictor utilities for the Credit Risk API.
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
    pass


class Predictor:
    def __init__(self) -> None:
        self.model = None
        self.features: Optional[List[str]] = None
        self.source: str = "none"
        self.n_features_expected: Optional[int] = None

    # ------------------------------------------------------------------
    # Model loading helpers
    # ------------------------------------------------------------------
    def _load_local_model(self, path: str):
        logger.info("Loading local model from %s", path)
        return joblib.load(path)

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
        MODEL_LOCAL_PATH = os.getenv("MODEL_LOCAL_PATH", "").strip()
        MODEL_FEATURES_PATH = os.getenv(
            "MODEL_FEATURES_PATH", "models/feature_names.json"
        ).strip()

        if not MODEL_LOCAL_PATH:
            raise ModelNotLoadedError(
                "MODEL_LOCAL_PATH must be set for local inference"
            )

        self.model = self._load_local_model(MODEL_LOCAL_PATH)
        self.source = f"local:{MODEL_LOCAL_PATH}"

        # ---------------------------
        # Infer expected feature schema
        # ---------------------------
        self.features = None
        self.n_features_expected = None

        # Preferred: explicit feature names
        if hasattr(self.model, "feature_names_in_"):
            self.features = list(self.model.feature_names_in_)
            self.n_features_expected = len(self.features)

        # Fallback: external feature list
        if self.features is None:
            self.features = self._load_feature_names_from_file(
                MODEL_FEATURES_PATH
            )
            if self.features:
                self.n_features_expected = len(self.features)

        # Last resort: feature count only
        if self.n_features_expected is None and hasattr(self.model, "n_features_in_"):
            self.n_features_expected = int(self.model.n_features_in_)

        if self.n_features_expected is None:
            raise RuntimeError(
                "Unable to determine expected number of input features"
            )

        logger.info(
            "Model ready | source=%s | feature_count=%s",
            self.source,
            self.n_features_expected,
        )

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------
    def _build_dataframe(self, features: Dict[str, Any]) -> pd.DataFrame:
        """
        Build a single-row DataFrame aligned to the model schema.
        """
        if self.features:
            row = {f: features.get(f, np.nan) for f in self.features}
            X = pd.DataFrame([row], columns=self.features)
        else:
            X = pd.DataFrame([features])

        if X.shape[1] != self.n_features_expected:
            raise ValueError(
                f"Input feature mismatch: got {X.shape[1]}, "
                f"expected {self.n_features_expected}"
            )

        return X

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None:
            raise ModelNotLoadedError("Model not loaded")

        X = self._build_dataframe(features)

        try:
            if hasattr(self.model, "predict_proba"):
                proba = float(self.model.predict_proba(X)[0, 1])
                predicted_class = int(proba >= 0.5)
            else:
                predicted_class = int(self.model.predict(X)[0])
                proba = None
        except Exception as exc:
            logger.exception("Prediction failed")
            raise

        return {
            "probability": proba,
            "predicted_class": predicted_class,
        }


# ----------------------------------------------------------------------
# Global singleton
# ----------------------------------------------------------------------
_predictor = Predictor()


def get_predictor() -> Predictor:
    if _predictor.model is None:
        _predictor.load()
    return _predictor
