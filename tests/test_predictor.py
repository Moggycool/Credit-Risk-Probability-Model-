import numpy as np
import pandas as pd
import pytest

from src.api.predictor import Predictor, ModelNotLoadedError


# -----------------------
# Mock model
# -----------------------
class MockModel:
    feature_names_in_ = ["f1", "f2", "f3"]

    def predict_proba(self, X):
        # return constant probability for testing
        return np.array([[0.3, 0.7]])

    def predict(self, X):
        return np.array([1])


# -----------------------
# Tests
# -----------------------
def test_predictor_predict_success(monkeypatch):
    """
    Predictor returns probability and class when model is loaded.
    """
    predictor = Predictor()
    predictor.model = MockModel()
    predictor.features = ["f1", "f2", "f3"]
    predictor.source = "local:test-model"

    result = predictor.predict({"f1": 1.0, "f2": 2.0, "f3": 3.0})

    assert result["probability"] == pytest.approx(0.7)
    assert result["predicted_class"] == 1


def test_predictor_missing_features_filled_with_nan():
    """
    Missing features should be filled with NaN.
    """
    predictor = Predictor()
    predictor.model = MockModel()
    predictor.features = ["f1", "f2", "f3"]

    X = predictor._build_dataframe({"f1": 1.0})

    assert X.shape == (1, 3)
    assert np.isnan(X.loc[0, "f2"])
    assert np.isnan(X.loc[0, "f3"])


def test_predictor_extra_features_ignored():
    """
    Extra features should not break prediction.
    """
    predictor = Predictor()
    predictor.model = MockModel()
    predictor.features = ["f1", "f2"]

    X = predictor._build_dataframe({"f1": 1.0, "f2": 2.0, "extra": 99})

    assert list(X.columns) == ["f1", "f2"]


def test_predictor_raises_if_model_not_loaded():
    """
    Predictor should raise if predict is called before load().
    """
    predictor = Predictor()

    with pytest.raises(ModelNotLoadedError):
        predictor.predict({"f1": 1.0})


def test_predictor_without_known_features():
    """
    If feature list is unknown, trust caller input.
    """
    predictor = Predictor()
    predictor.model = MockModel()
    predictor.features = None

    X = predictor._build_dataframe({"a": 1, "b": 2})

    assert isinstance(X, pd.DataFrame)
    assert list(X.columns) == ["a", "b"]
