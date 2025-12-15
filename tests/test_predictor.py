"""Tests for the Predictor class in api/predictor.py."""
from api.predictor import Predictor
import os

# Ensure env var is set BEFORE importing Predictor so module-level config sees it.
os.environ["MODEL_LOCAL_PATH"] = "models/random_forest_best.pkl"


def test_load_local_model_and_predict():
    """Test loading a local model and making a prediction."""
    p = Predictor()
    p.load()
    assert p.model is not None

    # Build a features dict. If Predictor inferred features, use those with dummy values.
    if p.features:
        features = {name: 0.0 for name in p.features}
    else:
        # Fallback: use an example feature name you expect in your model
        features = {"Amount_sum": 100.0}

    res = p.predict(features)
    assert "probability" in res
    # probability can be None if saved model only outputs labels, but predicted_class should exist
    assert "predicted_class" in res
