"""
Model loader and predictor utilities for the Credit Risk API.
"""

from typing import Any, Dict, List, Optional, Literal
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
        self.risk_thresholds: Dict[str, float] = {
            "low_max": 0.33,
            "medium_max": 0.66,
            "high_max": 1.0
        }

    # ------------------------------------------------------------------
    # Risk calculation helpers
    # ------------------------------------------------------------------
    def _calculate_risk_category(self, probability: float) -> str:
        """
        Convert probability to risk category.
        - LOW: 0.0 - 0.33 (0-33%)
        - MEDIUM: 0.33 - 0.66 (33-66%)
        - HIGH: 0.66 - 1.0 (66-100%)
        """
        if probability < self.risk_thresholds["low_max"]:
            return "LOW"
        elif probability < self.risk_thresholds["medium_max"]:
            return "MEDIUM"
        else:
            return "HIGH"

    def _get_risk_recommendation(self, risk_category: str) -> str:
        """
        Get business recommendation based on risk category.
        """
        recommendations = {
            "LOW": "Approve - Low risk of default",
            "MEDIUM": "Review - Moderate risk, consider additional verification",
            "HIGH": "Reject - High risk of default"
        }
        return recommendations.get(risk_category, "Review required")

    def _calculate_risk_score(self, probability: float) -> int:
        """
        Convert probability to a risk score (0-100).
        """
        return int(round(probability * 100, 0))

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
                    logger.info("Loaded %d feature names from %s", len(data), path)
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

        # Preferred: explicit feature names from the model
        if hasattr(self.model, "feature_names_in_"):
            self.features = list(self.model.feature_names_in_)
            self.n_features_expected = len(self.features)
            logger.info("Model provides %d feature names", self.n_features_expected)

        # Fallback: external feature list
        if self.features is None:
            self.features = self._load_feature_names_from_file(
                MODEL_FEATURES_PATH
            )
            if self.features:
                self.n_features_expected = len(self.features)
                logger.info("Loaded %d feature names from file",
                            self.n_features_expected)

        # Last resort: feature count only
        if self.n_features_expected is None and hasattr(self.model, "n_features_in_"):
            self.n_features_expected = int(self.model.n_features_in_)
            logger.info("Model expects %d features (no names)",
                        self.n_features_expected)

        if self.n_features_expected is None:
            raise RuntimeError(
                "Unable to determine expected number of input features"
            )

        if self.features:
            logger.info("Model expects features: %s", self.features)

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
        Raises ValueError if required features are missing.
        """
        if self.features:
            # Check for missing features BEFORE creating the dataframe
            missing_features = [f for f in self.features if f not in features]
            if missing_features:
                raise ValueError(
                    f"Missing required features: {missing_features}. "
                    f"Model expects exactly these features: {self.features}"
                )

            # Also check for extra features that will be ignored
            extra_features = [f for f in features.keys() if f not in self.features]
            if extra_features:
                logger.warning("Ignoring %d extra features: %s",
                               len(extra_features), extra_features[:5])

            # Create dataframe with only the expected features
            row = {f: features[f] for f in self.features}
            X = pd.DataFrame([row], columns=self.features)
        else:
            # Fallback: use whatever features are provided
            X = pd.DataFrame([features])
            logger.warning("No feature names available, using raw input features")

        if X.shape[1] != self.n_features_expected:
            raise ValueError(
                f"Input feature mismatch: got {X.shape[1]}, "
                f"expected {self.n_features_expected}. "
                f"Model expects: {self.features}"
            )

        # Check for NaN values that would cause the model to fail
        if X.isna().any().any():
            nan_features = X.columns[X.isna().any()].tolist()
            raise ValueError(
                f"Input contains NaN values for features: {nan_features}. "
                f"All features must have numeric values."
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
                risk_category = self._calculate_risk_category(proba)
                risk_score = self._calculate_risk_score(proba)
                recommendation = self._get_risk_recommendation(risk_category)
            else:
                predicted_class = int(self.model.predict(X)[0])
                proba = None
                risk_category = "HIGH" if predicted_class == 1 else "LOW"
                risk_score = 100 if predicted_class == 1 else 0
                recommendation = self._get_risk_recommendation(risk_category)
        except Exception as exc:
            logger.exception("Prediction failed: %s", exc)
            # Provide more helpful error messages
            error_msg = str(exc)
            if "NaN" in error_msg:
                raise ValueError(
                    "Model cannot handle NaN values. "
                    "Please ensure all feature values are provided."
                )
            raise

        return {
            "probability": proba,
            "predicted_class": predicted_class,
            "risk_category": risk_category,
            "risk_score": risk_score,
            "recommendation": recommendation,
            "model_source": self.source,
            "features_used": self.features,
        }


# ----------------------------------------------------------------------
# Global singleton
# ----------------------------------------------------------------------
_predictor = Predictor()


def get_predictor() -> Predictor:
    if _predictor.model is None:
        _predictor.load()
    return _predictor
