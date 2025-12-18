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

        # Overall aggregates
        agg_funcs = {
            self.amount_col: ["sum", "mean", "max", "std", "skew"],
            self.value_col:  ["sum", "mean", "max", "std", "skew"],
            self.time_col:   ["count", lambda x: (x.max() - x.min()).days + 1]
        }

        # Recent activity (last 30 days)
        latest_date = df[self.time_col].max()
        df_recent = df[df[self.time_col] >= (latest_date - pd.Timedelta(days=30))]
        recent_agg = df_recent.groupby(self.customer_id_col).agg({
            self.amount_col: ["sum", "mean"],
            self.value_col: ["sum", "mean"],
            self.time_col: "count"
        })

        recent_agg.columns = [f"Recent30_{c[0]}_{c[1]}" for c in recent_agg.columns]
        recent_agg = recent_agg.reset_index()

        # Overall aggregates
        overall_agg = df.groupby(self.customer_id_col).agg(agg_funcs)
        overall_agg.columns = [
            f"{c[0]}_{c[1] if isinstance(c[1], str) else 'period'}"
            for c in overall_agg.columns
        ]
        overall_agg = overall_agg.reset_index()

        # Merge recent and overall
        agg = pd.merge(overall_agg, recent_agg, on=self.customer_id_col, how='left')
        return agg


class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Generates temporal/cyclical features from transaction times at the per-customer level,
    including hour, weekday, day of the month, month, and year.
    Also includes velocity features and recency metrics.
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

        # Basic temporal features
        df["Hour"] = df[self.time_col].dt.hour
        df["Weekday"] = df[self.time_col].dt.weekday
        df["Day"] = df[self.time_col].dt.day
        df["Month"] = df[self.time_col].dt.month
        df["Year"] = df[self.time_col].dt.year
        df["IsWeekend"] = df["Weekday"].isin([5, 6]).astype(int)
        df["IsBusinessHours"] = df["Hour"].between(9, 17).astype(int)

        # Cyclical encoding for hour and month
        df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
        df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
        df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
        df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

        # Sort by customer and time for velocity calculations
        df = df.sort_values(by=[self.customer_id_col, self.time_col])

        # Time since last transaction
        df["TimeSinceLast"] = df.groupby(self.customer_id_col)[
            self.time_col].diff().dt.total_seconds() / 86400

        # Velocity features (transactions per time window)
        df["TransactionDate"] = df[self.time_col].dt.date

        # Calculate velocity per customer
        velocity_features = []
        for customer_id, group in df.groupby(self.customer_id_col):
            date_counts = group["TransactionDate"].value_counts().sort_index()

            # Rolling windows
            for window in [1, 7, 30]:
                rolling_avg = date_counts.rolling(window=window, min_periods=1).mean()
                group[f"Velocity_{window}d"] = rolling_avg.mean()

            velocity_features.append(group)

        df = pd.concat(velocity_features, ignore_index=True)

        # Aggregate to customer level
        aggs = df.groupby(self.customer_id_col).agg({
            "Hour": ["mean", "std"],
            "Weekday": ["mean", "std"],
            "Day": ["mean", self._first_mode_or_nan],
            "Month": ["mean", self._first_mode_or_nan],
            "Year": ["mean", self._first_mode_or_nan],
            "IsWeekend": "mean",
            "IsBusinessHours": "mean",
            "Hour_sin": "mean",
            "Hour_cos": "mean",
            "Month_sin": "mean",
            "Month_cos": "mean",
            "TimeSinceLast": ["mean", "min", "max"],
            "Velocity_1d": "mean",
            "Velocity_7d": "mean",
            "Velocity_30d": "mean"
        })

        # Flatten columns
        aggs.columns = ['_'.join([c[0], str(c[1]) if isinstance(c[1], str) else 'mode'])
                        for c in aggs.columns]
        aggs = aggs.reset_index()

        return aggs


class CategoryEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes low-cardinality categoricals using One-Hot encoding, high-cardinality as frequencies.

    Rationale:
    - One-hot encoding: Suitable for low-cardinality features (<8 categories) to avoid curse of dimensionality
    - Frequency encoding: Suitable for high-cardinality features to preserve information while reducing dimensionality
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
            # Build kwargs dynamically for sklearn compatibility
            sig = inspect.signature(OneHotEncoder.__init__)
            ohe_kwargs = {"handle_unknown": "ignore"}
            if 'sparse_output' in sig.parameters:
                ohe_kwargs['sparse_output'] = False
            elif 'sparse' in sig.parameters:
                ohe_kwargs['sparse'] = False
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

    Transformation Rationale:
    - log1p: Handles right-skewed distributions common in financial data
    - StandardScaler: Standardizes features for linear models
    - Correlation filtering: Reduces multicollinearity, improves model stability
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
            self.scaler = MinMaxScaler().fit(
                self._impute(fe_df[self.num_cols_after_fe])
            )
        return self

    def _impute(self, x_data):
        """Apply imputer or fillna(0) as per strategy."""
        if self.imputer is not None:
            return self.imputer.transform(x_data)
        else:
            result = pd.DataFrame(index=x.index)
        return result


class WoEIVCalculator(BaseEstimator, TransformerMixin):
    """
    Calculates Weight of Evidence (WoE) and Information Value (IV) for binary target.

    Usage:
    - Fits on customer-level data with binary target
    - Transforms customer-level aggregated features
    - Implements monotonic binning for continuous features
    - Supports regulatory interpretability requirements
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
                    # Use rank to avoid issues with repeated values
                    binned = pd.qcut(series.rank(method="first"),
                                     q=self.bins, duplicates='drop')
                except Exception:
                    binned = pd.cut(series, bins=self.bins)
                group = binned.astype(object)
            else:
                group = series.fillna("__NA__").astype(object)

            crosstab = pd.crosstab(group, y)
            # Ensure columns for good(0) and bad(1)
            if 0 not in crosstab.columns:
                crosstab[0] = 0
            if 1 not in crosstab.columns:
                crosstab[1] = 0
            crosstab = crosstab[[0, 1]].rename(columns={0: 'good', 1: 'bad'})
            # Replace zeros with a small number to avoid division by zero
            crosstab = crosstab.replace(0, 0.5)
            dist_good = crosstab['good'] / crosstab['good'].sum()
            dist_bad = crosstab['bad'] / crosstab['bad'].sum()
            woe = np.log(dist_good / dist_bad)
            iv = ((dist_good - dist_bad) * woe).sum()
            woe_map = woe.to_dict()
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
                continue
            woe_map = self.woe_maps[feature]
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
    encode_type: str = "onehot",
    scaler_type: str = "standard",
    impute_strategy: str = "mean",   # <-- Add this parameter
    categorical_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs all feature engineering steps and returns the enriched feature DataFrame and documentation.

    Pipeline Steps:
    1. Aggregate transaction features (overall + recent 30 days)
    2. Extract temporal features (cyclical, velocity, recency)
    3. Encode categorical features (one-hot for low-cardinality, frequency for high-cardinality)
    4. Normalize numeric features (log1p + standardization)
    5. Calculate WoE/IV for categorical features (if target available)
    6. Filter features based on IV thresholds for predictive power

    Returns:
    - feat_df: Customer-level features ready for modeling
    - feat_desc: Feature descriptions with transformations and IV values
    """

    df = raw_df.copy()
    # ---------------------------------------------------------
    # Create customer-level target for WoE (HasFraud)
    # ---------------------------------------------------------
    has_target = "FraudResult" in df.columns

    if has_target:
        df_target = (
            df.groupby(id_col)["FraudResult"]
              .max()
              .reset_index()
              .rename(columns={"FraudResult": "HasFraud"})
        )

        fraud_rate = df_target["HasFraud"].mean()
        print(f"Target summary — Fraud rate: {fraud_rate:.4%}")
    # Determine categorical columns if not provided
    if categorical_cols is None:
        auto_cats = df.select_dtypes(include=['object', 'category']).columns.drop([
            id_col, time_col], errors='ignore')
        categorical_cols = list(auto_cats)
    else:
        categorical_cols = [c for c in categorical_cols if c in df.columns]

    print("=== Feature Engineering Pipeline ===")
    print(f"Processing {len(df)} transactions for {df[id_col].nunique()} customers")
    print(f"Categorical columns: {categorical_cols}")

    # 1. Aggregate transaction features (overall + recent)
    print("1. Aggregating transaction features...")
    agg = TransactionAggregator(id_col, amount_col, value_col, time_col)
    df_agg = agg.fit_transform(df)

    # 2. Temporal features
    print("2. Extracting temporal features...")
    temp = TemporalFeatureEngineer(id_col, time_col)
    df_temp = temp.fit_transform(df)

    # 3. Customer-level categorical (mode per customer)
    print("3. Encoding categorical features...")
    if categorical_cols:
        df_cat = df[[id_col] + categorical_cols].copy()
        df_cat = df_cat.groupby(id_col, as_index=False).agg(
            lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
        if has_target:
            df_cat = df_cat.merge(df_target, on=id_col, how="left")
    else:
        df_cat = pd.DataFrame(
            {id_col: df_agg[id_col]}).drop_duplicates().reset_index(drop=True)

    # 3a. Fit CategoryEncoder on customer-level df_cat
    ce = CategoryEncoder(categorical_cols)
    df_cat_enc = pd.DataFrame({id_col: df_agg[id_col]})
    if categorical_cols and not df_cat.empty:
        ce.fit(df_cat)
        df_cat_enc = ce.transform(df_cat)
        if id_col not in df_cat_enc.columns and id_col in df_cat.columns:
            df_cat_enc[id_col] = df_cat[id_col].values
        df_cat_enc = df_cat_enc.reset_index(drop=True)

    # 4. Normalization & log transform for created numeric cols
    print("4. Normalizing numeric features...")
    numeric_prefixes = (f"{amount_col}", f"{value_col}", "Recent30")
    numeric_cols = [c for c in df_agg.columns if any(
        c.startswith(prefix) for prefix in numeric_prefixes) and 'std' not in c]

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

    # 6. WoE / IV encoding (customer-level, leakage-safe)
    # ---------------------------------------------------------
    ivs = {}
    woe_features_kept = []

    if has_target and categorical_cols:
        if df_cat["HasFraud"].nunique() < 2:
            print("⚠ WoE skipped: customer-level target has only one class.")
        else:
            print("5. Calculating WoE/IV for categorical features...")

            woe_features = [c for c in categorical_cols if c in df_cat.columns]

            woe_calc = WoEIVCalculator(
                features=woe_features,
                target_col="HasFraud",
                bins=4,               # safer for rare fraud
                min_bin_size=0.02
            )

            woe_calc.fit(df_cat)
            df_cat_woe = woe_calc.transform(df_cat)

            ivs = woe_calc.feature_iv()

            iv_threshold = 0.02
            high_iv_features = [f for f, iv in ivs.items() if iv >= iv_threshold]

            woe_features_kept = [
                f"{f}_woe" for f in high_iv_features
                if f"{f}_woe" in df_cat_woe.columns
            ]

            if woe_features_kept:
                feat_df = feat_df.merge(
                    df_cat_woe[[id_col] + woe_features_kept],
                    on=id_col,
                    how="left"
                )

            print(f"  WoE features kept: {woe_features_kept}")

    # 7. Feature description table
    print("6. Generating feature documentation...")
    desc = {"Feature": [], "Description": [], "Transformation": [], "IV_Value": []}

    for c in feat_df.columns:
        if c == id_col:
            continue

        iv_val = ""
        if c.endswith("_woe"):
            base = c[:-4]
            iv_val = f"{ivs.get(base, 0):.4f}" if base in ivs else ""
            desc["Feature"].append(c)
            desc["Description"].append(f"Weight of Evidence encoding for {base}")
            desc["Transformation"].append("WoE coding (monotonic binning)")
        elif c.startswith("Recent30_"):
            desc["Feature"].append(c)
            desc["Description"].append(f"Recent 30-day activity: {c}")
            desc["Transformation"].append("Windowed aggregation")
        elif "Velocity_" in c:
            window = c.split("_")[1].replace("d", "")
            desc["Feature"].append(c)
            desc["Description"].append(
                f"Transaction velocity (average per {window} days)")
            desc["Transformation"].append("Rolling window calculation")
        elif c in ["TimeSinceLast_mean", "TimeSinceLast_min", "TimeSinceLast_max"]:
            desc["Feature"].append(c)
            desc["Description"].append(
                f"Recency metric: {c.split('_')[-1]} days since last transaction")
            desc["Transformation"].append("Time difference aggregation")
        elif "Hour_sin" in c or "Hour_cos" in c or "Month_sin" in c or "Month_cos" in c:
            desc["Feature"].append(c)
            desc["Description"].append("Cyclical encoding for temporal patterns")
            desc["Transformation"].append("sin/cos transformation")
        elif "sum" in c:
            desc["Feature"].append(c)
            desc["Description"].append("Sum of values")
            desc["Transformation"].append("Customer aggregation")
        elif "mean" in c:
            desc["Feature"].append(c)
            desc["Description"].append("Mean value")
            desc["Transformation"].append("Customer aggregation")
        elif "skew" in c:
            desc["Feature"].append(c)
            desc["Description"].append("Skewness of values")
            desc["Transformation"].append("Customer aggregation")
        elif c.endswith("_log_std"):
            base = c.replace("_log_std", "")
            desc["Feature"].append(c)
            desc["Description"].append(f"Log-transformed and standardized {base}")
            desc["Transformation"].append("log1p + StandardScaler")
        elif categorical_cols and c in ce.get_feature_names():
            desc["Feature"].append(c)
            if "_freq" in c:
                desc["Description"].append(
                    f"Frequency encoding for {c.replace('_freq', '')}")
                desc["Transformation"].append("Frequency encoding")
            else:
                desc["Description"].append(f"One-hot encoding for {c}")
                desc["Transformation"].append("One-hot encoding")
        elif c.startswith("Hour_") or c.startswith("Weekday_"):
            desc["Feature"].append(c)
            desc["Description"].append("Temporal aggregation")
            desc["Transformation"].append("Extracted from timestamp, aggregated")
        elif c.startswith(("Day_", "Month_", "Year_")):
            desc["Feature"].append(c)
            desc["Description"].append("Calendar component aggregation")
            desc["Transformation"].append("Extracted from timestamp, aggregated")
        elif c in ["IsWeekend_mean", "IsBusinessHours_mean"]:
            desc["Feature"].append(c)
            desc["Description"].append(
                f"Fraction of transactions ({c.replace('_mean', '')})")
            desc["Transformation"].append("Boolean aggregation")
        else:
            desc["Feature"].append(c)
            desc["Description"].append("Generated feature")
            desc["Transformation"].append("See transformation details")

        desc["IV_Value"].append(iv_val)

    feat_desc = pd.DataFrame(desc)

    print(f"=== Pipeline Complete ===")
    print(f"Generated {feat_df.shape[1]} features for {feat_df.shape[0]} customers")
    print(f"Features with IV > 0.02: {len(woe_features_kept)}")

    return feat_df, feat_desc


if __name__ == "__main__":
    # Small smoke-test
    sample = pd.DataFrame([
        {"CustomerId": "C1", "Amount": 100.0, "Value": 10.0, "TransactionStartTime": "2025-01-01T12:00:00",
         "ProductCategory": "A", "ChannelId": "web", "FraudResult": 0},
        {"CustomerId": "C1", "Amount": 50.0, "Value": 5.0, "TransactionStartTime": "2025-01-02T13:00:00",
         "ProductCategory": "A", "ChannelId": "app", "FraudResult": 0},
        {"CustomerId": "C2", "Amount": 200.0, "Value": 20.0, "TransactionStartTime": "2025-01-01T10:00:00",
         "ProductCategory": "B", "ChannelId": "web", "FraudResult": 1},
        {"CustomerId": "C2", "Amount": 150.0, "Value": 15.0, "TransactionStartTime": "2025-01-03T14:00:00",
         "ProductCategory": "B", "ChannelId": "web", "FraudResult": 1},
    ])
    feat_df_smoke, feat_desc_smoke = feature_engineering_pipeline(
        sample, categorical_cols=["ProductCategory", "ChannelId"])
    print("Smoke test — features:", feat_df_smoke.shape, "desc:", feat_desc_smoke.shape)
    print("Feature columns:", feat_df_smoke.columns.tolist())
