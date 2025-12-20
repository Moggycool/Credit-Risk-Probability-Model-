"""
Integration tests for the Credit Risk API.
"""

import pytest
from fastapi.testclient import TestClient
import json
import os

from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Test the health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "timestamp" in data


def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Credit Risk Scoring API"
    assert "endpoints" in data


def test_model_info_endpoint():
    """Test the model info endpoint."""
    response = client.get("/model-info")
    assert response.status_code in [200, 503]  # 503 if model not loaded in test
    if response.status_code == 200:
        data = response.json()
        assert "model_source" in data
        assert "risk_categories" in data


def test_predict_endpoint_with_correct_features():
    """Test prediction with correct features."""
    payload = {
        "customer_id": "test_customer_001",
        "features": {
            "Year_mean": 2019.0,
            "Month_mean": 6.0
        }
    }

    response = client.post("/predict", json=payload)

    # Model might not be loaded in test environment
    if response.status_code == 200:
        data = response.json()
        assert "probability" in data
        assert "predicted_class" in data
        assert "risk_category" in data
        assert "risk_score" in data
        assert "recommendation" in data
        assert data["customer_id"] == "test_customer_001"

        # Validate probability range
        if data["probability"] is not None:
            assert 0 <= data["probability"] <= 1


def test_predict_endpoint_missing_features():
    """Test prediction with missing features."""
    payload = {
        "customer_id": "test_customer_002",
        "features": {
            "Year_mean": 2019.0
            # Missing Month_mean
        }
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 400  # Bad Request
    assert "Missing required features" in response.json()["detail"]


def test_predict_endpoint_wrong_features():
    """Test prediction with wrong features."""
    payload = {
        "customer_id": "test_customer_003",
        "features": {
            "wrong_feature_1": 100.0,
            "wrong_feature_2": 50.0
        }
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 400  # Bad Request


def test_predict_endpoint_without_customer_id():
    """Test prediction without customer_id."""
    payload = {
        "features": {
            "Year_mean": 2019.0,
            "Month_mean": 6.0
        }
    }

    response = client.post("/predict", json=payload)
    if response.status_code == 200:
        data = response.json()
        assert data["customer_id"] is None


def test_batch_example_endpoint():
    """Test the batch example endpoint."""
    response = client.get("/predict/batch-example")
    assert response.status_code == 200
    data = response.json()
    assert "examples" in data
    assert len(data["examples"]) > 0
    assert "feature_notes" in data


def test_swagger_docs():
    """Test that Swagger UI is accessible."""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_redoc_docs():
    """Test that ReDoc is accessible."""
    response = client.get("/redoc")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
