import sys
import pathlib
import pytest
from types import SimpleNamespace

# Ensure repo src/ is on sys.path so tests can import packages (api, src modules)
ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def mock_get_predictor(monkeypatch):
    """
    Autouse fixture that replaces api.predictor.get_predictor with a callable
    returning a fake predictor. Prevents real model loading at test time.
    """
    fake = SimpleNamespace()
    fake.source = "test:dummy"
    fake.features = None

    def _predict(features):
        return {"probability": 0.5, "predicted_class": 0}

    fake.predict = _predict

    # monkeypatch the get_predictor function used by the app
    monkeypatch.setattr("api.predictor.get_predictor", lambda: fake)
    # Replace the module-level _predictor singleton if api.predictor is importable
    try:
        import src.api.predictor as pred_mod  # noqa: E402
        pred_mod._predictor = fake  # type: ignore
    except Exception:
        pass

    yield
