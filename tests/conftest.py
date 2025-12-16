""" pytest fixture that mocks api.predictor.get_predictor globally so tests 
do not need model files.
"""
import pytest
from types import SimpleNamespace


@pytest.fixture(autouse=True)
def mock_get_predictor(monkeypatch):
    """
    Autouse fixture that replaces api.predictor.get_predictor with a callable
    returning a fake predictor. This prevents real model loading at test time
    and lets tests exercise endpoints / business logic in isolation.

    The fake predictor exposes:
      - .source (str)
      - .features (list[str] or None)
      - .predict(features_dict) -> {"probability": float, "predicted_class": int}
    """
    fake = SimpleNamespace()
    fake.source = "test:dummy"
    # minimal example features; tests that rely on feature names can monkeypatch this fixture
    fake.features = None

    def _predict(features):
        # deterministic dummy prediction for tests
        return {"probability": 0.5, "predicted_class": 0}

    fake.predict = _predict

    # monkeypatch the get_predictor function used by the app
    monkeypatch.setattr("api.predictor.get_predictor", lambda: fake)
    # Also ensure the module-level _predictor singleton, if referenced, is harmless:
    try:
        import importlib
        pred_mod = importlib.import_module("api.predictor")
        # replace the module-level _predictor with a prepared object to avoid side effects
        pred_mod._predictor = fake  # type: ignore
    except Exception:
        # non-fatal: if import fails here, tests that import api.predictor will still get monkeypatched get_predictor
        pass

    yield
