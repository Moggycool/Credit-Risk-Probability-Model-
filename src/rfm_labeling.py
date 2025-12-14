"""
RFMLabeler: compute Recency-Frequency-Monetary (RFM) and produce KMeans cluster labels.

Public class:
- RFMLabeler.fit(raw_df)                 # computes RFM and fits scaler+KMeans
- RFMLabeler.transform(customers_df)     # returns customers_df with cluster label
- RFMLabeler.fit_transform(raw_df)       # convenience: returns rfm_df with cluster
- Attributes: rfm_, scaler_, model_, cluster_profiles_, high_risk_label_
"""
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class RFMLabeler:
    def __init__(
        self,
        id_col: str = "CustomerId",
        time_col: str = "TransactionStartTime",
        amount_col: str = "Amount",
        snapshot_date: Optional[pd.Timestamp] = None,
        n_clusters: int = 3,
        random_state: int = 42,
        scale: bool = True,
    ):
        self.id_col = id_col
        self.time_col = time_col
        self.amount_col = amount_col
        self.snapshot_date = snapshot_date
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scale = scale

        # fitted attributes
        self.rfm_: Optional[pd.DataFrame] = None
        self.scaler_: Optional[StandardScaler] = None
        self.model_: Optional[KMeans] = None
        self.cluster_profiles_: Optional[pd.DataFrame] = None

    def _compute_rfm(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.time_col not in df.columns:
            raise KeyError(
                f"time_col '{self.time_col}' not found in DataFrame")
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        snapshot = pd.to_datetime(
            self.snapshot_date) if self.snapshot_date is not None else df[self.time_col].max() + pd.Timedelta(days=1)

        agg = (
            df.groupby(self.id_col)
            .agg(
                last_tx=(self.time_col, "max"),
                Recency=(self.time_col, lambda x: (snapshot - x.max()).days),
                Frequency=(self.time_col, "count"),
                Monetary=(self.amount_col, "sum"),
            )
        )
        # coerce types
        agg["Recency"] = agg["Recency"].astype(float)
        agg["Frequency"] = agg["Frequency"].astype(float)
        agg["Monetary"] = agg["Monetary"].astype(float)
        agg = agg.drop(columns=["last_tx"], errors="ignore")
        return agg

    def fit(self, raw_df: pd.DataFrame):
        """Compute RFM and fit scaler + KMeans on RFM features."""
        self.rfm_ = self._compute_rfm(raw_df)
        features = ["Recency", "Frequency", "Monetary"]
        X = self.rfm_[features].fillna(0).values

        if self.scale:
            self.scaler_ = StandardScaler()
            Xs = self.scaler_.fit_transform(X)
        else:
            Xs = X

        km = KMeans(n_clusters=self.n_clusters,
                    random_state=self.random_state, n_init=10)
        labels = km.fit_predict(Xs)

        self.rfm_ = self.rfm_.assign(cluster=labels)
        self.model_ = km
        self.cluster_profiles_ = self.rfm_.groupby(
            "cluster")[["Frequency", "Monetary", "Recency"]].mean()
        return self

    def transform(self, customers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Given customers_df containing id_col (column), return DataFrame
        with columns [id_col, cluster] for the customers present in rfm_.
        """
        if self.rfm_ is None:
            raise RuntimeError(
                "RFMLabeler.fit must be called before transform()")
        df_map = self.rfm_.reset_index()[[self.id_col, "cluster"]].copy()
        # If customers_df contains the id_col, preserve order/subset
        if self.id_col in customers_df.columns:
            result = customers_df[[self.id_col]].drop_duplicates().merge(
                df_map, on=self.id_col, how="left")
        else:
            # fallback: return mapping for all known customers
            result = df_map.copy()
        return result

    def fit_transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        self.fit(raw_df)
        return self.rfm_.reset_index()


# quick smoke test
if __name__ == "__main__":
    sample = pd.DataFrame(
        [
            {"CustomerId": "A", "Amount": 100, "TransactionStartTime": "2025-01-01"},
            {"CustomerId": "A", "Amount": 50, "TransactionStartTime": "2025-01-05"},
            {"CustomerId": "B", "Amount": 10, "TransactionStartTime": "2024-06-01"},
            {"CustomerId": "C", "Amount": 0, "TransactionStartTime": "2023-01-01"},
        ]
    )
    r = RFMLabeler()
    r.fit(sample)
    print(r.rfm_.head())
