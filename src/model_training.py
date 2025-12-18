"""
Task 5: Model Training, Selection, and Experiment Tracking

This module provides:
- Stratified train-test splitting
- Model training (Logistic Regression, Random Forest, Gradient Boosting)
- Hyperparameter tuning with cross-validation
- Evaluation metrics & visualizations
- MLflow experiment tracking
- Champion model selection

Author: (Your Name)
"""

import os
import json
import hashlib
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Dict, Tuple
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

RANDOM_STATE = 42
ARTIFACT_DIR = "artifacts"


# ---------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------

def hash_dataframe(df: pd.DataFrame) -> str:
    """Generate hash for dataset versioning."""
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()


def prepare_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Prepare stratified train-test split."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE
    )


def compute_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    """Compute classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


# ---------------------------------------------------------------------
# Visualization Functions
# ---------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_roc_curve(y_true, y_proba, path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(path)
    plt.close()


def plot_pr_curve(y_true, y_proba, path):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig(path)
    plt.close()


def plot_feature_importance(importances, path):
    importances.sort_values(ascending=False).head(20).plot(kind="bar")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# ---------------------------------------------------------------------
# Model Training
# ---------------------------------------------------------------------

def train_and_evaluate(
    X_train, X_test, y_train, y_test, model_name: str
) -> Dict:
    """Train, tune, evaluate, and log a model."""
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    if model_name == "logistic":
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000, class_weight="balanced"
            ))
        ])
        param_grid = {
            "model__C": [0.01, 0.1, 1, 10]
        }

    elif model_name == "random_forest":
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(random_state=RANDOM_STATE))
        ])
        param_grid = {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 5, 10]
        }

    elif model_name == "gradient_boosting":
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingClassifier(random_state=RANDOM_STATE))
        ])
        param_grid = {
            "model__learning_rate": [0.05, 0.1],
            "model__n_estimators": [100, 200]
        }

    else:
        raise ValueError("Unsupported model")

    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_proba)

    # Plots
    plot_confusion_matrix(
        y_test, y_pred, f"{ARTIFACT_DIR}/{model_name}_cm.png")
    plot_roc_curve(y_test, y_proba, f"{ARTIFACT_DIR}/{model_name}_roc.png")
    plot_pr_curve(y_test, y_proba, f"{ARTIFACT_DIR}/{model_name}_pr.png")

    # Feature importance
    if hasattr(best_model.named_steps["model"], "feature_importances_"):
        fi = pd.Series(
            best_model.named_steps["model"].feature_importances_,
            index=X_train.columns
        )
        fi_path = f"{ARTIFACT_DIR}/{model_name}_feature_importance.csv"
        fi.to_csv(fi_path)
        plot_feature_importance(fi, f"{ARTIFACT_DIR}/{model_name}_fi.png")
    else:
        fi_path = None

    return {
        "model": model_name,
        "best_estimator": best_model,
        "metrics": metrics,
        "best_params": search.best_params_,
        "feature_importance_path": fi_path
    }
