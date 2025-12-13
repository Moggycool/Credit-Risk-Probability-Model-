"""
Proxy Target Variable Engineering for Credit Risk Modeling
Calculates RFM features, clusters customers, assigns is_high_risk, and merges target for modeling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class ProxyTargetEngineer:
    def __init__(
        self,
        customer_id_col='CustomerId',
        date_col='TransactionStartTime',
        amount_col='Amount',
        n_clusters=3,
        random_state=42,
        scale_rfm=True,
        snapshot_date=None,
    ):
        self.customer_id_col = customer_id_col
        self.date_col = date_col
        self.amount_col = amount_col
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scale_rfm = scale_rfm
        self.snapshot_date = snapshot_date
        self.scaler = None
        self.kmeans = None
        self.cluster_centers_ = None

    def compute_rfm(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        if self.snapshot_date is None:
            snapshot_date = df[self.date_col].max() + pd.Timedelta(days=1)
        else:
            snapshot_date = pd.to_datetime(self.snapshot_date)
        recency = df.groupby(self.customer_id_col)[self.date_col].max().apply(
            lambda d: (snapshot_date - d).days)
        frequency = df.groupby(self.customer_id_col).size()
        monetary = df.groupby(self.customer_id_col)[self.amount_col].sum()
        rfm = pd.DataFrame(
            {'Recency': recency, 'Frequency': frequency, 'Monetary': monetary})
        rfm.index.name = self.customer_id_col
        return rfm

    def fit(self, df: pd.DataFrame):
        rfm = self.compute_rfm(df)
        X = rfm.values
        if self.scale_rfm:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        self.kmeans = KMeans(n_clusters=self.n_clusters,
                             random_state=self.random_state)
        labels = self.kmeans.fit_predict(X_scaled)
        self.cluster_centers_ = pd.DataFrame(
            self.kmeans.cluster_centers_, columns=rfm.columns)
        rfm['cluster'] = labels
        self.rfm_ = rfm  # stored for reference/summary
        return self

    def assign_high_risk(self, rfm: pd.DataFrame) -> pd.DataFrame:
        if self.cluster_centers_ is None:
            raise RuntimeError("fit() must be called first")
        high_risk_idx = self.cluster_centers_[
            'Recency'].idxmax()  # Highest recency = least active
        rfm['is_high_risk'] = (rfm['cluster'] == high_risk_idx).astype(int)
        return rfm

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        rfm_labeled = self.assign_high_risk(self.rfm_)
        return rfm_labeled

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        rfm = self.compute_rfm(df)
        X = rfm.values
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        clusters = self.kmeans.predict(X_scaled)
        rfm['cluster'] = clusters
        rfm = self.assign_high_risk(rfm)
        return rfm

    def merge_is_high_risk(self, features: pd.DataFrame, is_high_risk_labels: pd.DataFrame) -> pd.DataFrame:
        # Merge the is_high_risk column into feature dataframe
        if self.customer_id_col in features.columns:
            return features.merge(
                is_high_risk_labels[['is_high_risk']],
                left_on=self.customer_id_col, right_index=True, how='left'
            )
        else:
            # Assume CustomerId is index
            return features.merge(
                is_high_risk_labels[['is_high_risk']],
                left_index=True, right_index=True, how='left'
            )
