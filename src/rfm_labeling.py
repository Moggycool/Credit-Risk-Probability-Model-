"""
RFM proxy target variable engineering: calculates RFM features per customer,
segments with KMeans, and labels high-risk segment for credit risk proxy.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class RFMLabeler:
    """
    Computes RFM features and assigns 'is_high_risk' using KMeans clustering.
    """

    def __init__(
        self,
        customer_id_col='CustomerId',
        date_col='TransactionStartTime',
        amount_col='Amount',
        n_clusters=3,
        snapshot_date=None,
        random_state=42,
        scale_features=True,
    ):
        self.customer_id_col = customer_id_col
        self.date_col = date_col
        self.amount_col = amount_col
        self.n_clusters = n_clusters
        self.snapshot_date = snapshot_date
        self.random_state = random_state
        self.scale_features = scale_features
        self.scaler = None
        self.kmeans = None
        self.rfm_ = None

    def compute_rfm(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        if self.snapshot_date is None:
            snapshot_date = df[self.date_col].max() + pd.Timedelta(days=1)
        else:
            snapshot_date = pd.to_datetime(self.snapshot_date)
        # RFM
        recency = df.groupby(self.customer_id_col)[self.date_col].max().apply(
            lambda d: (snapshot_date - d).days)
        frequency = df.groupby(self.customer_id_col).size()
        monetary = df.groupby(self.customer_id_col)[self.amount_col].sum()
        rfm = pd.DataFrame({
            'Recency': recency,
            'Frequency': frequency,
            'Monetary': monetary,
        })
        rfm.index.name = self.customer_id_col
        return rfm

    def fit(self, df: pd.DataFrame):
        rfm = self.compute_rfm(df)
        X = rfm.values
        # Scale for clustering
        if self.scale_features:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        self.kmeans = KMeans(n_clusters=self.n_clusters,
                             random_state=self.random_state)
        cluster = self.kmeans.fit_predict(X_scaled)
        rfm['cluster'] = cluster
        self.rfm_ = rfm
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        rfm = self.compute_rfm(df)
        X = rfm.values
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        clusters = self.kmeans.predict(X_scaled)
        rfm['cluster'] = clusters
        # Identify high-risk cluster (max Recency, min Frequency & Monetary)
        centers = pd.DataFrame(self.kmeans.cluster_centers_, columns=[
                               'Recency', 'Frequency', 'Monetary'])
        # Highest recency = least recent
        risk_idx = centers['Recency'].idxmax()
        rfm['is_high_risk'] = (rfm['cluster'] == risk_idx).astype(int)
        return rfm[['Recency', 'Frequency', 'Monetary', 'cluster', 'is_high_risk']]

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.predict(df)
