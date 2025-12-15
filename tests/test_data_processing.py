"""Tests for data processing functions."""
from typing import Tuple, Dict, Any
from sklearn.dummy import DummyClassifier
import pandas as pd
import numpy as np
from src.model_training import prepare_data, evaluate_model, make_pipeline_with_scaler


def synthetic_df(n_customers=100, seed=0):
    """Generate a synthetic transactions DataFrame and aggregate to customer-level features."""
    # Use the new Generator API (default_rng) to avoid deprecation/type-checker warnings
    rng = np.random.default_rng(seed)
    ids = [f"C{i}" for i in range(n_customers)]
    rows = []
    for cid in ids:
        # Generator.integers is the equivalent of RandomState.randint
        n_tx = rng.integers(1, 5)
        for _ in range(n_tx):
            rows.append({
                "CustomerId": cid,
                "Amount": float(rng.exponential(scale=100.0)),
                "TransactionStartTime": "2025-01-01",
                # use Generator.random() instead of RandomState.rand()
                "is_high_risk": int(rng.random() < 0.2)
            })
    df = pd.DataFrame(rows)
    # build a simple customer-level features table (simulate pipeline output)
    feat = df.groupby("CustomerId").agg({"Amount": "sum"}).reset_index()
    feat["is_high_risk"] = df.groupby(
        "CustomerId")["is_high_risk"].first().values
    return feat


def test_prepare_data_shapes_and_reproducibility():
    df = synthetic_df(n_customers=50, seed=1)
    X_train1, X_test1, y_train1, y_test1 = prepare_data(
        df, test_size=0.3, random_state=123)
    X_train2, X_test2, y_train2, y_test2 = prepare_data(
        df, test_size=0.3, random_state=123)
    # shapes
    assert X_train1.shape[0] == y_train1.shape[0]
    assert X_test1.shape[0] == y_test1.shape[0]
    # reproducibility: splits identical for same random_state
    assert X_train1.reset_index(drop=True).equals(
        X_train2.reset_index(drop=True))
    assert X_test1.reset_index(drop=True).equals(
        X_test2.reset_index(drop=True))


def test_evaluate_model_returns_expected_keys():
    df = synthetic_df(n_customers=60, seed=2)
    X_train, X_test, y_train, y_test = prepare_data(
        df, test_size=0.2, random_state=2)
    # dummy classifier that predicts most frequent class
    dummy = DummyClassifier(strategy="most_frequent")
    pipe = make_pipeline_with_scaler(dummy)
    pipe.fit(X_train, y_train)
    metrics = evaluate_model(pipe, X_test, y_test)
    # expected metric keys present
    assert set(["accuracy", "precision", "recall", "f1",
               "roc_auc"]).issubset(set(metrics.keys()))
    # values numeric
    for v in metrics.values():
        assert isinstance(v, float)
