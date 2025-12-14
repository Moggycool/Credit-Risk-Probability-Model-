"""
Model training utilities: data preparation, model training/tuning, evaluation, MLflow logging.
"""
from typing import Dict, Tuple, Any, Optional
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings

# mlflow optional import
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True

except Exception:
    MLFLOW_AVAILABLE = False

warnings.filterwarnings("ignore")


def prepare_data(
    df: pd.DataFrame,
    id_col: str = "CustomerId",
    target_col: str = "is_high_risk",
    test_size: float = 0.2,
    random_state: int = 42,
    drop_columns: Optional[list] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare X_train, X_test, y_train, y_test. Keeps reproducibility via random_state.
    """
    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not present in DataFrame")

    drop_columns = drop_columns or [id_col]
    X = df.drop(columns=[
                target_col] + [c for c in drop_columns if c in df.columns], errors='ignore')
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() > 1 else None
    )
    return X_train, X_test, y_train, y_test


def build_default_models(random_state: int = 42) -> Dict[str, Any]:
    """Return a dict of candidate estimators (unwrapped)."""
    return {
        "logistic": LogisticRegression(solver="liblinear", random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "gbt": GradientBoostingClassifier(random_state=random_state)
    }


def default_param_grids() -> Dict[str, Dict]:
    """Parameter grids for Grid/Random search (example)."""
    return {
        "logistic": {
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__penalty": ["l1", "l2"]
        },
        "random_forest": {
            "clf__n_estimators": [50, 100],
            "clf__max_depth": [5, 10, None],
            "clf__min_samples_split": [2, 5]
        },
        "gbt": {
            "clf__n_estimators": [50, 100],
            "clf__learning_rate": [0.01, 0.1],
            "clf__max_depth": [3, 6]
        }
    }


def fit_and_tune(
    estimator,
    param_grid: Dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 3,
    search_type: str = "grid",
    n_iter: int = 20,
    scoring: str = "roc_auc",
    random_state: int = 42,
    n_jobs: int = -1
):
    """
    Fit and tune an estimator using GridSearchCV or RandomizedSearchCV (estimator should be a Pipeline).
    Returns fitted search object.
    """
    if search_type == "grid":
        search = GridSearchCV(estimator, param_grid, cv=cv,
                              scoring=scoring, n_jobs=n_jobs)
    else:
        search = RandomizedSearchCV(estimator, param_grid, n_iter=n_iter,
                                    cv=cv, scoring=scoring, random_state=random_state, n_jobs=n_jobs)
    search.fit(X_train, y_train)
    return search


def evaluate_model(estimator, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Compute standard evaluation metrics. estimator can be a pipeline."""
    y_pred = estimator.predict(X_test)
    y_proba = None
    try:
        if hasattr(estimator, "predict_proba"):
            y_proba = estimator.predict_proba(X_test)[:, 1]
        elif hasattr(estimator, "decision_function"):
            y_proba = estimator.decision_function(X_test)
    except Exception:
        y_proba = None

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0))
    }
    if y_proba is not None and len(np.unique(y_test)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
        except Exception:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def make_pipeline_with_scaler(clf):
    """
    Build a standard Pipeline with a StandardScaler followed by the classifier.
    Useful for classifiers that need scaling (e.g., logistic).
    """
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def log_experiment_mlflow(
    name: str,
    estimator,
    params: Dict,
    metrics: Dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    artifacts: Optional[Dict[str, str]] = None,
    model_save_path: Optional[str] = None,
    register_name: Optional[str] = None
) -> Optional[str]:
    """
    Log experiment to MLflow. Returns run_id or None if mlflow unavailable.
    """
    if not MLFLOW_AVAILABLE:
        print("MLflow not available; skipping mlflow logging.")
        return None

    mlflow.set_experiment(name)
    with mlflow.start_run() as run:
        mlflow.log_params(params or {})
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))
        # log artifacts (files)
        if artifacts:
            for tag, path in artifacts.items():
                if os.path.exists(path):
                    mlflow.log_artifact(path, artifact_path=tag)

        # Save and log model artifact
        if model_save_path:
            mlflow.log_artifact(model_save_path, artifact_path="model")
            try:
                mlflow.sklearn.log_model(estimator, "model")
            except Exception:
                pass

        run_id = run.info.run_id

        # Optionally register
        if register_name:
            try:
                model_uri = f"runs:/{run_id}/model"
                mlflow.register_model(model_uri, register_name)
            except Exception as exc:
                print(
                    "Model registration failed (likely local tracking without registry):", exc)
        return run_id


def save_model_local(estimator, path: str):
    """Save model to local filesystem with joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(estimator, path)
    return path
