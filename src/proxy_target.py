"""
ProxyTargetEngineer: wraps RFMLabeler to produce a binary proxy target is_high_risk
and merge it back into a customer-level DataFrame.

Public class:
- ProxyTargetEngineer.fit(raw_df)                   # fits internal RFMLabeler
- ProxyTargetEngineer.assign_labels(feat_df)        # returns feat_df with is_high_risk column
- ProxyTargetEngineer.fit_transform(raw_df, feat_df) # convenience
"""
from typing import Optional, Dict
import pandas as pd
import numpy as np

# Adjust import path depending on your package layout. If this module sits in src/,
# and you run notebooks with src/ on sys.path, the following import works.
from rfm_labeling import RFMLabeler  # use relative import if packaged


class ProxyTargetEngineer:
    def __init__(
        self,
        id_col: str = "CustomerId",
        time_col: str = "TransactionStartTime",
        amount_col: str = "Amount",
        snapshot_date: Optional[pd.Timestamp] = None,
        n_clusters: int = 3,
        random_state: int = 42,
        scale: bool = True,
        selection_method: str = "freq_mon_combined_rank",
    ):
        self.id_col = id_col
        self.time_col = time_col
        self.amount_col = amount_col
        self.snapshot_date = snapshot_date
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scale = scale
        self.selection_method = selection_method

        self.rfm_labeler: Optional[RFMLabeler] = None
        self.high_risk_label: Optional[int] = None

    def fit(self, raw_df: pd.DataFrame):
        """Fit the internal RFMLabeler and pick the high-risk cluster label."""
        self.rfm_labeler = RFMLabeler(
            id_col=self.id_col,
            time_col=self.time_col,
            amount_col=self.amount_col,
            snapshot_date=self.snapshot_date,
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            scale=self.scale,
        ).fit(raw_df)

        # determine high-risk cluster according to selection_method
        profiles = self.rfm_labeler.cluster_profiles_.copy()
        # rank by Frequency and Monetary (lower -> higher risk)
        profiles["freq_rank"] = profiles["Frequency"].rank(
            method="dense", ascending=True)
        profiles["mon_rank"] = profiles["Monetary"].rank(
            method="dense", ascending=True)
        profiles["combined_rank"] = profiles["freq_rank"] + \
            profiles["mon_rank"]
        self.high_risk_label = int(profiles["combined_rank"].idxmin())
        return self

    def assign_labels(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge is_high_risk into feat_df. Returns a copy with 'is_high_risk' (0/1).
        Customers not present in RFM (e.g., no tx) receive is_high_risk=0 by default.
        """
        if self.rfm_labeler is None or self.high_risk_label is None:
            raise RuntimeError(
                "ProxyTargetEngineer.fit must be called before assign_labels()")

        rfm_map = self.rfm_labeler.rfm_.reset_index(
        )[[self.id_col, "cluster"]].copy()
        rfm_map["is_high_risk"] = (
            rfm_map["cluster"] == self.high_risk_label).astype(int)
        # merge into feat_df
        result = feat_df.merge(
            rfm_map[[self.id_col, "is_high_risk"]], on=self.id_col, how="left")
        result["is_high_risk"] = result["is_high_risk"].fillna(0).astype(int)
        return result

    def fit_transform(self, raw_df: pd.DataFrame, feat_df: pd.DataFrame) -> pd.DataFrame:
        self.fit(raw_df)
        return self.assign_labels(feat_df)


# smoke test
if __name__ == "__main__":
    import pandas as pd
    sample = pd.DataFrame(
        [
            {"CustomerId": "A", "Amount": 100, "TransactionStartTime": "2025-01-01"},
            {"CustomerId": "A", "Amount": 50, "TransactionStartTime": "2025-01-05"},
            {"CustomerId": "B", "Amount": 10, "TransactionStartTime": "2024-06-01"},
        ]
    )
    proxy = ProxyTargetEngineer()
    proxy.fit(sample)
    df_feat = pd.DataFrame({"CustomerId": ["A", "B", "C"]})
    print(proxy.assign_labels(df_feat))
