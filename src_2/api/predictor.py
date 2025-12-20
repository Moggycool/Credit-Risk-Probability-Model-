"""
Model loader and predictor utilities for the Credit Risk API.
Supports both local models and MLflow Model Registry.
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
    # Model loading helpers
    # ------------------------------------------------------------------
    def _load_local_model(self, path: str):
        """Load model from local file (joblib or pickle)."""
        logger.info("Loading local model from %s", path)
        try:
            import warnings
            from sklearn.exceptions import InconsistentVersionWarning

            # Suppress the version warning
            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

            model = joblib.load(path)

            # Restore warnings
            warnings.filterwarnings("default", category=InconsistentVersionWarning)

            return model

        except Exception as e:
            logger.error("Failed to load model with joblib: %s", e)
            # Try pickle as fallback
            try:
                import pickle
                with open(path, 'rb') as f:
                    model = pickle.load(f)
                logger.info("Loaded model with pickle instead")
                return model
            except Exception as e2:
                logger.error("Failed to load model with pickle: %s", e2)
                raise
        # return joblib.load(path)

    def _load_mlflow_model(self, model_uri: str):
        """Load model from MLflow Model Registry."""
        try:
            import mlflow.pyfunc
            logger.info("Loading MLflow model from %s", model_uri)
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info("Successfully loaded MLflow model: %s", model_uri)
            return model
        except ImportError:
            logger.error("MLflow not installed. Install with: pip install mlflow")
            raise
        except Exception as exc:
            logger.error("Failed to load MLflow model: %s", exc)
            raise

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
        MODEL_MLFLOW_URI = os.getenv("MODEL_MLFLOW_URI", "").strip()
        MODEL_FEATURES_PATH = os.getenv(
            "MODEL_FEATURES_PATH", "models/feature_names.json"
        ).strip()

        if not MODEL_LOCAL_PATH and not MODEL_MLFLOW_URI:
            raise ModelNotLoadedError(
                "Either MODEL_LOCAL_PATH or MODEL_MLFLOW_URI must be set"
            )

        # Load from MLflow Model Registry if URI is provided
        if MODEL_MLFLOW_URI:
            self.model = self._load_mlflow_model(MODEL_MLFLOW_URI)
            self.source = f"mlflow:{MODEL_MLFLOW_URI}"
        else:
            # Fallback to local model
            self.model = self._load_local_model(MODEL_LOCAL_PATH)
            self.source = f"local:{MODEL_LOCAL_PATH}"

        # ---------------------------
        # Infer expected feature schema
        # ---------------------------
        self.features = None
        self.n_features_expected = None

        # Try to get features from MLflow model first
        if hasattr(self.model, 'metadata') and hasattr(self.model.metadata, 'get_input_schema'):
            try:
                input_schema = self.model.metadata.get_input_schema()
                if input_schema:
                    self.features = [field.name for field in input_schema.inputs]
                    self.n_features_expected = len(self.features)
                    logger.info("Extracted %d features from MLflow model schema",
                                self.n_features_expected)
            except Exception as e:
                logger.warning("Could not extract features from MLflow schema: %s", e)

        # Fallback: explicit feature names from the model
        if not self.features and hasattr(self.model, "feature_names_in_"):
            self.features = list(self.model.feature_names_in_)
            self.n_features_expected = len(self.features)
            logger.info("Model provides %d feature names", self.n_features_expected)

        # Fallback: external feature list
        if not self.features:
            self.features = self._load_feature_names_from_file(MODEL_FEATURES_PATH)
            if self.features:
                self.n_features_expected = len(self.features)
                logger.info("Loaded %d feature names from file",
                            self.n_features_expected)

        # Last resort: feature count only
        if self.n_features_expected is None and hasattr(self.model, "n_features_in_"):
            self.n_features_expected = int(self.model.n_features_in_)
            logger.info("Model expects %d features (no names)",
                        self.n_features_expected)

        # For MLflow models, check if we can extract feature info differently
        if self.n_features_expected is None and hasattr(self.model, '_model_impl'):
            try:
                # Try to get sklearn model from MLflow wrapper
                sklearn_model = self.model._model_impl
                if hasattr(sklearn_model, 'n_features_in_'):
                    self.n_features_expected = int(sklearn_model.n_features_in_)
                    logger.info("Extracted feature count from MLflow model impl")
            except:
                pass

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
    # Risk calculation helpers
    # ------------------------------------------------------------------
    def _calculate_risk_category(self, probability: float) -> str:
        """Convert probability to risk category."""
        if probability < self.risk_thresholds["low_max"]:
            return "LOW"
        elif probability < self.risk_thresholds["medium_max"]:
            return "MEDIUM"
        else:
            return "HIGH"

    def _get_risk_recommendation(self, risk_category: str) -> str:
        """Get business recommendation based on risk category."""
        recommendations = {
            "LOW": "Approve - Low risk of default",
            "MEDIUM": "Review - Moderate risk, consider additional verification",
            "HIGH": "Reject - High risk of default"
        }
        return recommendations.get(risk_category, "Review required")

    def _calculate_risk_score(self, probability: float) -> int:
        """Convert probability to a risk score (0-100)."""
        return int(round(probability * 100, 0))

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------
    def _build_dataframe(self, features: Dict[str, Any]) -> pd.DataFrame:
        """Build a single-row DataFrame aligned to the model schema."""
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
            # MLflow models have a predict method
            if hasattr(self.model, 'predict'):
                # For MLflow models
                prediction_result = self.model.predict(X)

                # Handle different MLflow model output formats
                if isinstance(prediction_result, np.ndarray):
                    if prediction_result.ndim == 2 and prediction_result.shape[1] == 2:
                        # MLflow model returning probabilities
                        proba = float(prediction_result[0, 1])
                    elif prediction_result.ndim == 1:
                        # MLflow model returning classes
                        proba = float(prediction_result[0])
                    else:
                        proba = float(prediction_result[0])
                else:
                    # Fallback for other formats
                    proba = float(prediction_result)

                predicted_class = int(proba >= 0.5)

            elif hasattr(self.model, "predict_proba"):
                # For sklearn models
                proba = float(self.model.predict_proba(X)[0, 1])
                predicted_class = int(proba >= 0.5)
            else:
                # Fallback
                predicted_class = int(self.model.predict(X)[0])
                proba = None

            # Calculate risk metrics
            risk_category = self._calculate_risk_category(
                proba) if proba is not None else "UNKNOWN"
            risk_score = self._calculate_risk_score(proba) if proba is not None else 0
            recommendation = self._get_risk_recommendation(risk_category)

        except Exception as exc:
            logger.exception("Prediction failed: %s", exc)
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
