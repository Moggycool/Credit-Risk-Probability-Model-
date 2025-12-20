from types import SimpleNamespace
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Test the /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"


def test_predict_endpoint(monkeypatch):
    """Test the /predict endpoint with correct features."""

    # ✅ realistic fake predictor that matches your actual model
    fake_predictor = SimpleNamespace(
        model=True,  # must exist
        features=["Year_mean", "Month_mean"],  # Actual features your model expects
        source="local:models/logistic_champion_fixed.joblib",
        predict=lambda features: {
            "probability": 0.3331,  # Example probability for Year=2019, Month=6
            "predicted_class": 0,    # Example class
        },
    )

    # monkeypatch get_predictor used by main.py
    monkeypatch.setattr(
        "src.api.main.get_predictor",
        lambda: fake_predictor,
    )

    # Test with correct features that your model actually expects
    payload = {
        "customer_id": "test-001",
        "features": {
            "Year_mean": 2019.0,
            "Month_mean": 6.0,
        },
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()

    assert body["probability"] == 0.3331
    assert body["predicted_class"] == 0
    assert body["customer_id"] == "test-001"
    assert body["model"] == "local:models/logistic_champion_fixed.joblib"


def test_predict_missing_features(monkeypatch):
    """Test that missing features return appropriate error."""

    fake_predictor = SimpleNamespace(
        model=True,
        features=["Year_mean", "Month_mean"],
        source="test-model",
        predict=lambda features: {"probability": 0.5, "predicted_class": 0}
    )

    monkeypatch.setattr(
        "src.api.main.get_predictor",
        lambda: fake_predictor,
    )

    # Test with missing Month_mean
    payload = {
        "customer_id": "test-002",
        "features": {
            "Year_mean": 2019.0,
            # Missing Month_mean
        },
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 400  # Bad Request


def test_predict_wrong_features(monkeypatch):
    """Test that wrong features return appropriate error."""

    fake_predictor = SimpleNamespace(
        model=True,
        features=["Year_mean", "Month_mean"],
        source="test-model",
        predict=lambda features: {"probability": 0.5, "predicted_class": 0}
    )

    monkeypatch.setattr(
        "src.api.main.get_predictor",
        lambda: fake_predictor,
    )

    # Test with wrong features
    payload = {
        "customer_id": "test-003",
        "features": {
            "Wrong_feature_1": 100.0,
            "Wrong_feature_2": 50.0,
        },
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 400  # Bad Request


def test_predict_no_customer_id(monkeypatch):
    """Test prediction without customer_id."""

    fake_predictor = SimpleNamespace(
        model=True,
        features=["Year_mean", "Month_mean"],
        source="test-model",
        predict=lambda features: {
            "probability": 0.7,
            "predicted_class": 1,
        },
    )

    monkeypatch.setattr(
        "src.api.main.get_predictor",
        lambda: fake_predictor,
    )

    payload = {
        # No customer_id
        "features": {
            "Year_mean": 2019.0,
            "Month_mean": 12.0,
        },
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["probability"] == 0.7
    assert body["predicted_class"] == 1
    assert body["customer_id"] is None
