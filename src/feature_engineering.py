"""Feature engineering module for customer transaction data."""

from typing import Optional, List
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom feature engineering transformer for customer transaction data.
    Performs customer-level aggregations, extraction of time features, 
    encoding, imputation, scaling, and optionally Weight of Evidence (WoE).
    """

    def __init__(
        # pylint: disable=too-many-arguments
        self,
        customer_id_col: str = "CustomerId",
        time_col: str = "TransactionStartTime",
        amount_col: str = "Amount",
        value_col: str = "Value",
        encode_type: str = "onehot",     # "onehot" or "label"
        scaler_type: str = "standard",   # "standard" or "minmax"
        impute_strategy: str = "mean",   # "mean", "median", or "zero"
        categorical_cols: Optional[List[str]] = None,
        woe_cols: Optional[List[str]] = None,
    ):
        self.customer_id_col = customer_id_col
        self.time_col = time_col
        self.amount_col = amount_col
        self.value_col = value_col
        self.encode_type = encode_type
        self.scaler_type = scaler_type
        self.categorical_cols = categorical_cols
        self.woe_cols = woe_cols
        self.impute_strategy = impute_strategy
        self.ohe = None
        self.lbl_encoders = None
        self.scaler = None
        self.imputer = None
        self.num_cols_after_fe = None

    def _engineer_features(self, df):
        # Aggregation
        agg_df = df.groupby(self.customer_id_col).agg({
            self.amount_col: ['sum', 'mean', 'count', 'std'],
            self.value_col: ['sum', 'mean', 'std']
        })
        agg_df.columns = [f"{col[0]}_{col[1]}" for col in agg_df.columns]
        agg_df = agg_df.reset_index()
        # Time features
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        for part in ['hour', 'day', 'month', 'year']:
            df[f"transaction_{part}"] = getattr(df[self.time_col].dt, part)
        mode_features = df.groupby(self.customer_id_col).agg({
            'transaction_hour': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
            'transaction_day': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
            'transaction_month': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
            'transaction_year': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
        }).reset_index()
        out = pd.merge(agg_df, mode_features,
                       on=self.customer_id_col, how='left')
        # Latest categorical values
        if self.categorical_cols is None:
            self.categorical_cols = df.select_dtypes(
                include='object').columns.tolist()
        cat_df = df.sort_values(self.time_col).groupby(self.customer_id_col).tail(1)[
            [self.customer_id_col] + self.categorical_cols]
        cat_df = cat_df.set_index(self.customer_id_col)
        if self.encode_type == 'onehot' and self.ohe is not None:
            enc = self.ohe.transform(cat_df[self.categorical_cols])
            cat_df = pd.DataFrame(
                enc, index=cat_df.index, columns=self.ohe.get_feature_names_out(self.categorical_cols))
        elif self.encode_type == 'label' and self.lbl_encoders is not None:
            for col in self.categorical_cols:
                cat_df[col] = self.lbl_encoders[col].transform(
                    cat_df[col].astype(str))
        out = pd.merge(out, cat_df, left_on=self.customer_id_col,
                       right_index=True, how='left')
        return out

    def fit(self, x_data: pd.DataFrame, y=None):
        """Fit the encoders, imputer, and scaler on the engineered features."""
        # reference y to avoid "unused argument" warnings from linters
        if y is not None:
            _ = y
        df = x_data.copy()
        if self.categorical_cols is None:
            self.categorical_cols = df.select_dtypes(
                include='object').columns.tolist()
        last_cats = df.sort_values(self.time_col).groupby(
            self.customer_id_col).tail(1)
        if self.encode_type == 'onehot':
            self.ohe = OneHotEncoder(
                handle_unknown='ignore', sparse_output=False)
            self.ohe.fit(last_cats[self.categorical_cols])
        elif self.encode_type == 'label':
            self.lbl_encoders = {col: LabelEncoder().fit(
                last_cats[col].astype(str)) for col in self.categorical_cols}
        fe_df = self._engineer_features(df)
        self.num_cols_after_fe = [col for col in fe_df.columns if pd.api.types.is_numeric_dtype(
            fe_df[col]) and col != self.customer_id_col]
        # Imputer
        if self.impute_strategy == "zero":
            self.imputer = None
        else:
            self.imputer = SimpleImputer(strategy=self.impute_strategy)
            self.imputer.fit(fe_df[self.num_cols_after_fe])
        # Scaler
        if self.scaler_type == "standard":
            self.scaler = StandardScaler().fit(
                self._impute(fe_df[self.num_cols_after_fe])
            )
        else:
            self.scaler = MinMaxScaler().fit(
                self._impute(fe_df[self.num_cols_after_fe])
            )
        return self

    def _impute(self, x_data):
        """Apply imputer or fillna(0) as per strategy."""
        if self.imputer is not None:
            return self.imputer.transform(x_data)
        else:
            return x_data.fillna(0).to_numpy()

    def transform(self, x_data: pd.DataFrame, y=None):
        """Transform the data using engineered features and fitted scaler."""
        fe_df = self._engineer_features(x_data)
        # Impute numeric columns
        numeric_values = self._impute(fe_df[self.num_cols_after_fe])
        # Scale
        fe_df[self.num_cols_after_fe] = self.scaler.transform(numeric_values)
        fe_df = fe_df.fillna(0)
        return fe_df

    def add_weight_of_evidence(self, df: pd.DataFrame, target_col: str):
        """Add Weight of Evidence (WoE) transformed features using xverse."""
        try:
            from xverse.transformer import WOE
        except ImportError as exc:
            raise ImportError(
                "WoE transformation requires xverse library. Install via: pip install xverse"
            ) from exc
        if not self.woe_cols:
            raise ValueError("Specify 'woe_cols' to use WoE transformation.")
        woe = WOE()
        woe.fit(df[self.woe_cols], df[target_col])
        transformed = woe.transform(df[self.woe_cols])
        df = pd.concat([df, transformed], axis=1)
        return df


def build_pipeline(
    # pylint: disable=too-many-arguments
    customer_id_col: str = "CustomerId",
    time_col: str = "TransactionStartTime",
    amount_col: str = "Amount",
    value_col: str = "Value",
    encode_type: str = "onehot",
    scaler_type: str = "standard",
    impute_strategy: str = "mean",   # <-- Add this parameter
    categorical_cols: Optional[List[str]] = None
) -> Pipeline:
    """Construct a feature engineering pipeline for customer transaction data."""
    fe = FeatureEngineer(
        customer_id_col=customer_id_col,
        time_col=time_col,
        amount_col=amount_col,
        value_col=value_col,
        encode_type=encode_type,
        scaler_type=scaler_type,
        impute_strategy=impute_strategy,    # <-- Pass to FeatureEngineer
        categorical_cols=categorical_cols
    )
    pipeline = Pipeline([
        ("feature_engineer", fe)
    ])
    return pipeline
