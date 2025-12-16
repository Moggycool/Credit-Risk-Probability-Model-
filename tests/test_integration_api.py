"""Integration tests for the API endpoints."""
# tests/test_integration_api.py
import json
import requests


def test_predict_smoke():
    """Smoke test for the /predict endpoint."""
    with open("tests/fixtures/payload_smoke_test.json", "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    r = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=10)
    assert r.status_code == 200, f"{r.status_code} {r.text}"
    j = r.json()
    assert "probability" in j
    assert "predicted_class" in j
