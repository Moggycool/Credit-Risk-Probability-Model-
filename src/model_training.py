"""
Model training utilities: data preparation, model training/tuning, evaluation, MLflow logging.

This module improves robustness for saving models so that relative paths are saved under the
repository root `models/` directory (found automatically). It also provides helpers to
save an MLflow-style model folder locally (if mlflow is available).
"""
from typing import Dict, Tuple, Any, Optional
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings
import os
import shutil

# mlflow optional import
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False

warnings.filterwarnings("ignore")


# Utilities
def find_repo_root(start: Path = Path.cwd(), markers=("src", ".git", "requirements.txt")) -> Path:
    """
    Walk up from `start` to find a likely repository root (contains one of markers).
    If not found, return the original start path.
    """
    p = start.resolve()
    for _ in range(8):
        if any((p / m).exists() for m in markers):
            return p
        if p.parent == p:
            break
        p = p.parent
    return start.resolve()


def _resolve_target_path(path: str) -> Path:
    """
    Resolve a user-supplied path string to an absolute Path inside the repo.
    If `path` is absolute, it is returned unchanged. If relative, it is interpreted
    relative to the repository root and returned.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    repo_root = find_repo_root()
    return (repo_root / p).resolve()


# Core functionality (unchanged behavior, slightly hardened)
def prepare_data(
    df: pd.DataFrame,
    id_col: str = "CustomerId",
    target_col: str = "is_high_risk",
    test_size: float = 0.2,
    random_state: int = 42,
    drop_columns: Optional[list] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not present in DataFrame")
    drop_columns = drop_columns or [id_col]
    X = df.drop(columns=[target_col] +
                [c for c in drop_columns if c in df.columns], errors='ignore')
    y = df[target_col].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if y.nunique() > 1 else None
    )
    return X_train, X_test, y_train, y_test


def build_default_models(random_state: int = 42) -> Dict[str, Any]:
    return {
        "logistic": LogisticRegression(solver="liblinear", random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "gbt": GradientBoostingClassifier(random_state=random_state)
    }


def default_param_grids() -> Dict[str, Dict]:
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
    if search_type == "grid":
        search = GridSearchCV(estimator, param_grid, cv=cv,
                              scoring=scoring, n_jobs=n_jobs)
    else:
        search = RandomizedSearchCV(estimator, param_grid, n_iter=n_iter, cv=cv,
                                    scoring=scoring, random_state=random_state, n_jobs=n_jobs)
    search.fit(X_train, y_train)
    return search


def evaluate_model(estimator, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
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
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", clf)
    ])


def save_model_local(estimator, path: str) -> str:
    """
    Save model to local filesystem with joblib.

    If `path` is relative, it will be resolved relative to the repository root
    (the function will create the parent directories as needed). Returns the
    absolute path string of the saved file.
    """
    target = _resolve_target_path(path)
    os.makedirs(target.parent, exist_ok=True)
    joblib.dump(estimator, str(target))
    return str(target)


def save_mlflow_model_folder(estimator, path: str) -> Optional[str]:
    """
    Save an MLflow-style model folder using mlflow.sklearn.save_model if mlflow is available.
    `path` may be relative (resolved against repo root) or absolute.
    Returns the path string of the saved folder, or None if mlflow not available or an error occurred.
    """
    target = _resolve_target_path(path)
    try:
        if not MLFLOW_AVAILABLE:
            print("mlflow not available; skipping save_mlflow_model_folder")
            return None
        # ensure parent exists and remove existing folder to avoid stale content
        if target.exists():
            if target.is_file():
                target.unlink()
            else:
                shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        mlflow.sklearn.save_model(sk_model=estimator, path=str(target))
        return str(target)
    except Exception as exc:
        print("Failed to save mlflow model folder:", exc)
        return None


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

    Behavior improvements:
    - If model_save_path is provided and is relative, it will be resolved under repo-root (so callers
      can pass the same relative path they used with save_model_local).
    - The function will attempt to log the provided artifact files and the model folder (if possible).
    """
    if not MLFLOW_AVAILABLE:
        print("MLflow not available; skipping mlflow logging.")
        return None

    mlflow.set_experiment(name)
    with mlflow.start_run() as run:
        mlflow.log_params(params or {})
        for k, v in metrics.items():
            try:
                mlflow.log_metric(k, float(v))
            except Exception:
                # skip non-floatable metrics
                pass

        # log artifacts (files) after resolving relative paths
        if artifacts:
            for tag, path in artifacts.items():
                try:
                    resolved = _resolve_target_path(path)
                    if resolved.exists():
                        mlflow.log_artifact(str(resolved), artifact_path=tag)
                    else:
                        # try as-is (maybe remote path)
                        if os.path.exists(path):
                            mlflow.log_artifact(path, artifact_path=tag)
                except Exception as exc:
                    print(f"Failed to log artifact {path}: {exc}")

        # Save and log model artifact
        run_id = run.info.run_id
        if model_save_path:
            resolved_model_path = _resolve_target_path(model_save_path)
            # If a file exists at resolved_model_path, upload it to artifacts/model
            try:
                if resolved_model_path.exists():
                    mlflow.log_artifact(str(resolved_model_path), artifact_path="model")
                # Also try to save a MLflow model folder under ./models/mlflow_model_<run_id>
                mlflow_model_folder = Path("models") / f"mlflow_model_{run_id}"
                mlflow_model_folder_path = save_mlflow_model_folder(
                    estimator, str(mlflow_model_folder))
                if mlflow_model_folder_path:
                    # log the saved model folder as artifacts
                    mlflow.log_artifacts(mlflow_model_folder_path,
                                         artifact_path="model")
                # Attempt to log using mlflow.sklearn.log_model (may fail on non-sklearn objects)
                try:
                    mlflow.sklearn.log_model(estimator, "model")
                except Exception:
                    pass
            except Exception as exc:
                print("Error while saving/logging model to mlflow:", exc)

        # Optionally register
        if register_name:
            try:
                model_uri = f"runs:/{run_id}/model"
                mlflow.register_model(model_uri, register_name)
            except Exception as exc:
                print("Model registration failed (likely local tracking without registry):", exc)

        return run_id


# Exported alias kept for compatibility
def save_model_local_legacy(estimator, path: str) -> str:
    """Backwards-compatible alias."""
    return save_model_local(estimator, path)
