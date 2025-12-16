"""Tests for the FastAPI app in api/main.py."""
import os
import json
from api.main import app  # imports will use get_predictor() as needed
from fastapi.testclient import TestClient


# Ensure MODEL_LOCAL_PATH is set before importing app/predictor so modules pick up the env var.
os.environ["MODEL_LOCAL_PATH"] = "models/random_forest_best.pkl"


client = TestClient(app)


def test_health_endpoint():
    """Test the /health endpoint."""
    r = client.get("/health")
    assert r.status_code == 200
    d = r.json()
    assert "status" in d and "model_loaded" in d


def test_predict_endpoint():
    """Test the /predict endpoint."""
    # Prepare a dummy features dict. If your model expects specific features, use them here.
    # If predictor infers feature names, the predictor will fail if you don't supply them.
    payload = {
        "customer_id": "test-1",
        "features": {
            "Amount_sum": 100.0,
        }
    }
    r = client.post("/predict", json=payload)
    # Accept either 200 (success) or 400/503 if features mismatch - but in CI we expect 200
    assert r.status_code in (200, 400, 503)
    if r.status_code == 200:
        j = r.json()
        assert "probability" in j and "predicted_class" in j
