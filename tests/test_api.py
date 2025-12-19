from types import SimpleNamespace
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_predict_endpoint(monkeypatch):
    """Test the /predict endpoint."""

    # ✅ realistic fake predictor
    fake_predictor = SimpleNamespace(
        model=True,  # must exist
        features=["Amount_sum"],
        source="test-model",
        predict=lambda features: {
            "probability": 0.7,
            "predicted_class": 1,
        },
    )

    # monkeypatch get_predictor used by main.py
    monkeypatch.setattr(
        "src.api.main.get_predictor",
        lambda: fake_predictor,
    )

    payload = {
        "customer_id": "test-1",
        "features": {
            "Amount_sum": 100.0,
        },
    }

    r = client.post("/predict", json=payload)

    assert r.status_code == 200
    body = r.json()

    assert body["probability"] == 0.7
    assert body["predicted_class"] == 1
    assert body["customer_id"] == "test-1"
