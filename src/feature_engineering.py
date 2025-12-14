"""
Feature Engineering Module for Credit Risk Modeling.

Provides:
- TransactionAggregator: Aggregates transaction data into customer-level features.
- TemporalFeatureEngineer: Extracts temporal/cyclical features.
- CategoryEncoder: Encodes categorical features appropriately.
- FeatureNormalizer: Handles log-transformation and scaling.
- WoEIVCalculator: Calculates Weight of Evidence (WoE) and Information Value (IV) for binary targets.
- feature_engineering_pipeline: Executes all steps and returns engineered DataFrame and documentation.
"""

from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
import inspect
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = [
    "TransactionAggregator",
    "TemporalFeatureEngineer",
    "CategoryEncoder",
    "FeatureNormalizer",
    "WoEIVCalculator",
    "feature_engineering_pipeline"
]


class TransactionAggregator(BaseEstimator, TransformerMixin):
    """Aggregates raw transaction data to customer-level features."""

    def __init__(
        self,
        customer_id_col: str = "CustomerId",
        amount_col: str = "Amount",
        value_col: str = "Value",
        time_col: str = "TransactionStartTime"
    ):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
        self.value_col = value_col
        self.time_col = time_col

    def fit(self, x, y=None):
        _ = x
        _ = y
        return self

    def transform(self, x):
        df = x.copy()
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        agg_funcs = {
            self.amount_col: ["sum", "mean", "max", "std", "skew"],
            self.value_col:  ["sum", "mean", "max", "std", "skew"],
            self.time_col:   ["count", lambda x: (x.max() - x.min()).days + 1]
        }
        agg = df.groupby(self.customer_id_col).agg(agg_funcs)
        agg.columns = [
            f"{c[0]}_{c[1] if isinstance(c[1], str) else 'period'}"
            for c in agg.columns
        ]
        agg = agg.reset_index()
        return agg


class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Generates temporal/cyclical features from transaction times at the per-customer level,
    including hour, weekday, day of the month, month, and year.

    NOTE: mode aggregations return scalars (first mode) to avoid list-like cells.
    """

    def __init__(
        self, customer_id_col="CustomerId",
        time_col="TransactionStartTime"
    ):
        self.customer_id_col = customer_id_col
        self.time_col = time_col

    def fit(self, x, y=None):
        _ = x
        _ = y
        return self

    def _first_mode_or_nan(self, s: pd.Series):
        m = s.mode()
        return m.iloc[0] if not m.empty else np.nan

    def transform(self, x):
        df = x.copy()
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        df["Hour"] = df[self.time_col].dt.hour
        df["Weekday"] = df[self.time_col].dt.weekday
        df["Day"] = df[self.time_col].dt.day
        df["Month"] = df[self.time_col].dt.month
        df["Year"] = df[self.time_col].dt.year
        df["IsWeekend"] = df["Weekday"].isin([5, 6]).astype(int)

        aggs = df.groupby(self.customer_id_col).agg({
            "Hour": ["mean", "std"],
            "Weekday": ["mean", "std"],
            "Day": ["mean", self._first_mode_or_nan],
            "Month": ["mean", self._first_mode_or_nan],
            "Year": ["mean", self._first_mode_or_nan],
            "IsWeekend": "mean",
        })

        # flatten columns
        aggs.columns = ['_'.join([c[0], str(c[1]) if isinstance(
            c[1], str) else 'mode']) for c in aggs.columns]
        aggs = aggs.reset_index()
        return aggs


class CategoryEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes low-cardinality categoricals using One-Hot encoding, high-cardinality as frequencies.
    """

    def __init__(self, categorical_columns: List[str], low_card_thr: int = 8):
        self.categorical_columns = categorical_columns or []
        self.low_card_thr = low_card_thr
        self.ohe: Optional[OneHotEncoder] = None
        self.high_card_cols: List[str] = []
        self.low_card_cols: List[str] = []
        self._feature_names: List[str] = []

    def fit(self, x: pd.DataFrame, y=None):
        low_card = []
        high_card = []
        for col in self.categorical_columns:
            if col not in x.columns:
                continue
            n_unique = x[col].nunique(dropna=False)
            if n_unique <= self.low_card_thr:
                low_card.append(col)
            else:
                high_card.append(col)
        self.low_card_cols = low_card
        self.high_card_cols = high_card

        if self.low_card_cols:
            # Build kwargs dynamically so static analyzers (pylint) do not
            # complain about passing a deprecated/removed keyword like
            # `sparse` or `sparse_output` directly. We inspect the
            # constructor signature at runtime and then pass supported args.
            sig = inspect.signature(OneHotEncoder.__init__)
            ohe_kwargs = {"handle_unknown": "ignore"}
            if 'sparse_output' in sig.parameters:
                ohe_kwargs['sparse_output'] = False
            elif 'sparse' in sig.parameters:
                ohe_kwargs['sparse'] = False
            # instantiate with dynamic kwargs to avoid static lint errors
            self.ohe = OneHotEncoder(**ohe_kwargs)
            self.ohe.fit(x[self.low_card_cols].astype(object).fillna("__NA__"))
            self._feature_names = list(
                self.ohe.get_feature_names_out(self.low_card_cols))
        else:
            self.ohe = None
            self._feature_names = []

        return self

    def transform(self, x: pd.DataFrame):
        dfs = []
        if self.low_card_cols and self.ohe is not None:
            arr = self.ohe.transform(
                x[self.low_card_cols].astype(object).fillna("__NA__"))
            df_ohe = pd.DataFrame(
                arr, columns=self._feature_names, index=x.index)
            dfs.append(df_ohe)

        for col in self.high_card_cols:
            if col in x.columns:
                freqs = x[col].fillna("__NA__").value_counts(normalize=True)
                df_freq = x[[col]].fillna("__NA__").replace(
                    freqs).rename(columns={col: f"{col}_freq"})
                dfs.append(df_freq)

        result = pd.concat(dfs, axis=1) if dfs else pd.DataFrame(index=x.index)
        result.index = x.index
        return result

    def get_feature_names(self):
        return self._feature_names + [f"{col}_freq" for col in self.high_card_cols]


class FeatureNormalizer(BaseEstimator, TransformerMixin):
    """
    Applies log transformation and standard scaling to numeric columns.
    Drops a column if highly collinear (correlation > corr_thr).
    """

    def __init__(self, numeric_cols: List[str], drop_collinear: bool = True, corr_thr=0.95):
        self.numeric_cols = numeric_cols or []
        self.drop_collinear = drop_collinear
        self.corr_thr = corr_thr
        self.scaler = StandardScaler()
        self.cols_to_use = list(self.numeric_cols)

    def fit(self, x: pd.DataFrame, y=None):
        if not self.numeric_cols:
            return self
        x_num = x[self.numeric_cols].copy()
        x_log = x_num.apply(np.log1p)
        if self.drop_collinear and len(self.numeric_cols) > 1:
            corr = x_log.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            drop = [column for column in upper.columns if any(
                upper[column] > self.corr_thr)]
            self.cols_to_use = [
                col for col in self.numeric_cols if col not in drop]
        else:
            self.cols_to_use = list(self.numeric_cols)
        if self.cols_to_use:
            self.scaler.fit(x_log[self.cols_to_use])
        return self

    def transform(self, x: pd.DataFrame):
        if not self.numeric_cols:
            return pd.DataFrame(index=x.index)
        x_num = x[self.numeric_cols].copy()
        x_log = x_num.apply(np.log1p)
        x_use = x_log[self.cols_to_use] if self.cols_to_use else pd.DataFrame(
            index=x.index)
        if not x_use.empty:
            arr = self.scaler.transform(x_use)
            result = pd.DataFrame(
                arr, columns=[f"{c}_log_std" for c in self.cols_to_use], index=x.index)
        else:
            result = pd.DataFrame(index=x.index)
        return result


class WoEIVCalculator(BaseEstimator, TransformerMixin):
    """
    Calculates Weight of Evidence (WoE) and Information Value (IV) for binary target.
    Fits on a DataFrame that contains raw features and target at the same granularity
    (e.g., transaction-level), but can transform another DataFrame that contains the
    same categorical columns (e.g., customer-level aggregated df_cat).
    """

    def __init__(self, features: List[str], target_col: str, bins: int = 5, min_bin_size: float = 0.01):
        self.features = features or []
        self.target_col = target_col
        self.bins = bins
        self.min_bin_size = min_bin_size
        self.woe_maps = {}
        self.ivs = {}

    def fit(self, x: pd.DataFrame, y=None):
        df = x.copy()
        if self.target_col not in df.columns:
            raise ValueError(
                f"Target column '{self.target_col}' not found in provided DataFrame for WoE fit.")
        y = df[self.target_col]
        for feature in self.features:
            if feature not in df.columns:
                continue
            series = df[feature]
            # Decide grouping: numeric -> bin, categorical -> as-is
            if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=False) > self.bins:
                try:
                    # use rank to avoid issues with repeated values
                    binned = pd.qcut(series.rank(method="first"),
                                     q=self.bins, duplicates='drop')
                except Exception:
                    binned = pd.cut(series, bins=self.bins)
                group = binned.astype(object)
            else:
                group = series.fillna("__NA__").astype(object)

            crosstab = pd.crosstab(group, y)
            # ensure columns for good(0) and bad(1)
            if 0 not in crosstab.columns:
                crosstab[0] = 0
            if 1 not in crosstab.columns:
                crosstab[1] = 0
            crosstab = crosstab[[0, 1]].rename(columns={0: 'good', 1: 'bad'})
            # replace zeros with a small number to avoid division by zero
            crosstab = crosstab.replace(0, 0.5)
            dist_good = crosstab['good'] / crosstab['good'].sum()
            dist_bad = crosstab['bad'] / crosstab['bad'].sum()
            woe = np.log(dist_good / dist_bad)
            iv = ((dist_good - dist_bad) * woe).sum()
            woe_map = woe.to_dict()
            # for binned numeric, keys will be Interval objects; keep them as-is for mapping
            self.woe_maps[feature] = woe_map
            self.ivs[feature] = iv
        return self

    def transform(self, x: pd.DataFrame):
        x_ = x.copy()
        for feature in self.features:
            if feature not in x_.columns:
                continue
            series = x_[feature]
            if feature not in self.woe_maps:
                # no mapping learned for this feature
                continue
            woe_map = self.woe_maps[feature]
            # if numeric and mapping is based on bins, bin accordingly
            if pd.api.types.is_numeric_dtype(series) and any(not isinstance(k, str) and hasattr(k, "left") for k in woe_map.keys()):
                try:
                    bins = pd.qcut(series.rank(method="first"),
                                   q=len(woe_map), duplicates='drop')
                except Exception:
                    bins = pd.cut(series, bins=len(woe_map))
                x_[f"{feature}_woe"] = bins.map(woe_map).fillna(0)
            else:
                x_[f"{feature}_woe"] = series.fillna(
                    "__NA__").map(woe_map).fillna(0)
        return x_

    def feature_iv(self):
        return self.ivs


def feature_engineering_pipeline(
    raw_df: pd.DataFrame,
    id_col: str = "CustomerId",
    amount_col: str = "Amount",
    value_col: str = "Value",
    time_col: str = "TransactionStartTime",
    categorical_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs all feature engineering steps and returns the enriched
    feature DataFrame and description table.
    """

    df = raw_df.copy()
    # determine categorical columns if not provided
    if categorical_cols is None:
        auto_cats = df.select_dtypes(include=['object', 'category']).columns.drop([
            id_col, time_col], errors='ignore')
        categorical_cols = list(auto_cats)
    else:
        # keep only columns that actually exist in raw_df
        categorical_cols = [c for c in categorical_cols if c in df.columns]

    # 1. Aggregate transaction features
    agg = TransactionAggregator(id_col, amount_col, value_col, time_col)
    df_agg = agg.fit_transform(df)

    # 2. Temporal features
    temp = TemporalFeatureEngineer(id_col, time_col)
    df_temp = temp.fit_transform(df)

    # 3. Customer-level categorical (mode per customer)
    if categorical_cols:
        df_cat = df[[id_col] + categorical_cols].copy()
        # take the first mode per group or the first value if no mode
        df_cat = df_cat.groupby(id_col, as_index=False).agg(
            lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
    else:
        # ensure df_cat has id column for merges
        df_cat = pd.DataFrame(
            {id_col: df_agg[id_col]}).drop_duplicates().reset_index(drop=True)

    # 3a. Fit CategoryEncoder on the customer-level df_cat (if any categorical cols)
    ce = CategoryEncoder(categorical_cols)
    df_cat_enc = pd.DataFrame({id_col: df_agg[id_col]})
    if categorical_cols and not df_cat.empty:
        ce.fit(df_cat)
        df_cat_enc = ce.transform(df_cat)
        # ensure id column present for merge
        if id_col not in df_cat_enc.columns:
            # df_cat likely still has id_col in df_cat; bring it in
            if id_col in df_cat.columns:
                df_cat_enc[id_col] = df_cat[id_col].values
            else:
                df_cat_enc[id_col] = df_cat.index
        df_cat_enc = df_cat_enc.reset_index(drop=True)

    # 4. Normalization & log transform for created numeric cols
    numeric_prefixes = (f"{amount_col}", f"{value_col}")
    numeric_cols = [c for c in df_agg.columns if c.startswith(
        numeric_prefixes) and 'std' not in c]
    fn = FeatureNormalizer(numeric_cols) if numeric_cols else None
    if fn:
        fn.fit(df_agg)
        df_norm = fn.transform(df_agg)
        df_norm[id_col] = df_agg[id_col].values
    else:
        df_norm = pd.DataFrame({id_col: df_agg[id_col]})

    # 5. Combine all features (so far)
    feat_df = df_agg.merge(df_temp, on=id_col, how='left')
    feat_df = feat_df.merge(df_cat_enc, on=id_col, how='left')
    feat_df = feat_df.merge(df_norm, on=id_col, how='left')

    # 6. Optional: WoE/IV encoding - compute on customer-level categorical df_cat (which contains raw categories per customer)
    ivs = {}
    if 'FraudResult' in df.columns and categorical_cols:
        woe_features = [c for c in categorical_cols if c in df_cat.columns]
        if woe_features:
            woe_calc = WoEIVCalculator(
                woe_features, target_col='FraudResult', bins=5)
            # fit using transaction-level data WITH target
            woe_calc.fit(df)
            # transform the customer-level categorical table (make sure id_col present)
            df_cat_for_woe = df_cat.reset_index(drop=True)
            df_cat_woe = woe_calc.transform(df_cat_for_woe)
            # ensure id present
            if id_col not in df_cat_woe.columns and id_col in df_cat_for_woe.columns:
                df_cat_woe[id_col] = df_cat_for_woe[id_col].values
            woe_cols_to_merge = [
                f"{c}_woe" for c in woe_features if f"{c}_woe" in df_cat_woe.columns]
            if woe_cols_to_merge and id_col in df_cat_woe.columns:
                feat_df = feat_df.merge(
                    df_cat_woe[[id_col] + woe_cols_to_merge], on=id_col, how='left')
            ivs = woe_calc.feature_iv()

    # 7. Feature description table
    desc = {"Feature": [], "Description": [], "Transformation": []}
    for c in feat_df.columns:
        if c == id_col:
            continue
        if c.endswith("_woe"):
            base = c[:-4]
            iv_str = f" (IV={ivs[base]:.3f})" if base in ivs else ""
            desc["Feature"].append(c)
            desc["Description"].append(
                f"Weight of Evidence encoding for {base} wrt FraudResult{iv_str}")
            desc["Transformation"].append("WoE coding (see IV)")
        elif "sum" in c:
            desc["Feature"].append(c)
            desc["Description"].append("Sum of values")
            desc["Transformation"].append("Customer aggregation, none")
        elif "mean" in c:
            desc["Feature"].append(c)
            desc["Description"].append("Mean value")
            desc["Transformation"].append("Customer aggregation, none")
        elif "skew" in c:
            desc["Feature"].append(c)
            desc["Description"].append("Skewness of values")
            desc["Transformation"].append("Customer aggregation, none")
        elif c.endswith("_log_std"):
            desc["Feature"].append(c)
            desc["Description"].append(
                "Log-transformed and normalized numeric feature")
            desc["Transformation"].append("log1p & StandardScaler")
        elif categorical_cols and c in ce.get_feature_names():
            desc["Feature"].append(c)
            desc["Description"].append(f"Encoded {c.split('_')[0]}")
            desc["Transformation"].append("One-hot or frequency encoding")
        elif c.startswith("Hour_") or c.startswith("Weekday_"):
            desc["Feature"].append(c)
            desc["Description"].append("Cyclical or time-aggregation")
            desc["Transformation"].append(
                "Extracted from timestamp, aggregated")
        elif c.startswith(("Day_", "Month_", "Year_")):
            desc["Feature"].append(c)
            desc["Description"].append(
                "Calendar/date component (customer aggregation)")
            desc["Transformation"].append(
                "Extracted from timestamp, aggregated")
        elif c == "IsWeekend_mean":
            desc["Feature"].append(c)
            desc["Description"].append("Fraction of transactions on weekends")
            desc["Transformation"].append("Extracted/aggregated")
        else:
            desc["Feature"].append(c)
            desc["Description"].append("Generated feature")
            desc["Transformation"].append("See code")

    feat_desc = pd.DataFrame(desc)
    return feat_df, feat_desc


_ = feature_engineering_pipeline


if __name__ == "__main__":
    # Small smoke-test to exercise feature_engineering_pipeline
    sample = pd.DataFrame([
        {"CustomerId": "C1", "Amount": 100.0, "Value": 10.0, "TransactionStartTime": "2025-01-01T12:00:00",
         "ProductCategory": "A", "ChannelId": "web", "FraudResult": 0},
        {"CustomerId": "C1", "Amount": 50.0, "Value": 5.0, "TransactionStartTime": "2025-01-02T13:00:00",
         "ProductCategory": "A", "ChannelId": "app", "FraudResult": 0},
        {"CustomerId": "C2", "Amount": 200.0, "Value": 20.0, "TransactionStartTime": "2025-01-01T10:00:00",
         "ProductCategory": "B", "ChannelId": "web", "FraudResult": 1},
    ])
    feat_df_smoke, feat_desc_smoke = feature_engineering_pipeline(
        sample, categorical_cols=["ProductCategory", "ChannelId"])
    print("Smoke test — features:", feat_df_smoke.shape,
          "desc:", feat_desc_smoke.shape)
