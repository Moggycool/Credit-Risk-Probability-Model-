"""
Unit tests for the Predictor class.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import os

from src.api.predictor import Predictor, ModelNotLoadedError


def test_predictor_initialization():
    """Test Predictor initialization."""
    predictor = Predictor()
    assert predictor.model is None
    assert predictor.features is None
    assert predictor.source == "none"
    assert predictor.n_features_expected is None
    assert "low_max" in predictor.risk_thresholds


def test_calculate_risk_category():
    """Test risk category calculation."""
    predictor = Predictor()

    assert predictor._calculate_risk_category(0.1) == "LOW"
    assert predictor._calculate_risk_category(0.32) == "LOW"
    assert predictor._calculate_risk_category(0.33) == "MEDIUM"
    assert predictor._calculate_risk_category(0.5) == "MEDIUM"
    assert predictor._calculate_risk_category(0.65) == "MEDIUM"
    assert predictor._calculate_risk_category(0.66) == "HIGH"
    assert predictor._calculate_risk_category(0.9) == "HIGH"


def test_get_risk_recommendation():
    """Test risk recommendation."""
    predictor = Predictor()

    assert "Approve" in predictor._get_risk_recommendation("LOW")
    assert "Review" in predictor._get_risk_recommendation("MEDIUM")
    assert "Reject" in predictor._get_risk_recommendation("HIGH")


def test_calculate_risk_score():
    """Test risk score calculation."""
    predictor = Predictor()

    assert predictor._calculate_risk_score(0.0) == 0
    assert predictor._calculate_risk_score(0.5) == 50
    assert predictor._calculate_risk_score(1.0) == 100
    assert predictor._calculate_risk_score(0.333) == 33


@patch('src.api.predictor.joblib.load')
def test_load_local_model(mock_joblib_load):
    """Test loading local model."""
    predictor = Predictor()
    mock_model = Mock()
    mock_joblib_load.return_value = mock_model

    # Temporarily set environment variable
    os.environ["MODEL_LOCAL_PATH"] = "dummy/path/model.joblib"
    os.environ["MODEL_FEATURES_PATH"] = "dummy/path/features.json"

    try:
        predictor.load()
        assert predictor.model == mock_model
        assert predictor.source.startswith("local:")
    finally:
        # Clean up
        del os.environ["MODEL_LOCAL_PATH"]
        del os.environ["MODEL_FEATURES_PATH"]


def test_build_dataframe_with_features():
    """Test dataframe building with known features."""
    predictor = Predictor()
    predictor.features = ["feature1", "feature2"]
    predictor.n_features_expected = 2

    features = {"feature1": 10.5, "feature2": 20.0}
    df = predictor._build_dataframe(features)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 2)
    assert list(df.columns) == ["feature1", "feature2"]
    assert df.iloc[0]["feature1"] == 10.5


def test_build_dataframe_missing_features():
    """Test dataframe building with missing features."""
    predictor = Predictor()
    predictor.features = ["feature1", "feature2", "feature3"]
    predictor.n_features_expected = 3

    features = {"feature1": 10.5, "feature2": 20.0}
    # Missing feature3

    with pytest.raises(ValueError, match="Missing required features"):
        predictor._build_dataframe(features)


def test_predict_without_model():
    """Test prediction without loaded model."""
    predictor = Predictor()

    with pytest.raises(ModelNotLoadedError):
        predictor.predict({"feature1": 10.5})


def test_singleton_pattern():
    """Test that get_predictor returns a singleton."""
    from src.api.predictor import get_predictor, _predictor

    predictor1 = get_predictor()
    predictor2 = get_predictor()

    # Should be the same instance
    assert predictor1 is predictor2
    assert predictor1 is _predictor
