import pandas as pd
from src.model_training import prepare_data


def test_target_column_exists():
    df = pd.read_csv("data/processed/features_with_target.csv")
    assert "is_high_risk" in df.columns


def test_stratified_split():
    df = pd.read_csv("data/processed/features_with_target.csv")
    X_train, X_test, y_train, y_test = prepare_data(df, "is_high_risk")
    assert abs(y_train.mean() - y_test.mean()) < 0.02
