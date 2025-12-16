# tests/conftest.py
import sys
from pathlib import Path

# Insert project src/ at front of sys.path so tests can import project modules
ROOT = Path(__file__).resolve().parents[1]  # repo root
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
