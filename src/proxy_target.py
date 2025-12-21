"""
Proxy Target Variable Engineering Module (Task 4)

Objective:
Create a supervised proxy target (is_high_risk) using RFM analysis
and K-Means clustering, to enable downstream modeling when direct
fraud/default labels are unavailable or extremely sparse.

Responsibilities:
- RFM feature calculation at customer level
- Behavioral segmentation using K-Means (n=3)
- High-risk cluster identification
- Proxy label assignment
- Merge proxy target with engineered features
- Audit-friendly outputs
"""

from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# =========================================================
# 1. RFM CALCULATION
# =========================================================
def calculate_rfm(
    transactions: pd.DataFrame,
    customer_id_col: str = "CustomerId",
    amount_col: str = "Amount",
    time_col: str = "TransactionStartTime",
    snapshot_date: pd.Timestamp | None = None
) -> pd.DataFrame:
    """
    Calculate RFM metrics per customer.

    Recency   : Days since last transaction (lower = better)
    Frequency : Number of transactions
    Monetary  : Total transaction amount

    A consistent snapshot date is enforced for recency.
    """

    df = transactions.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    if snapshot_date is None:
        snapshot_date = df[time_col].max()

    rfm = (
        df.groupby(customer_id_col)
        .agg(
            Recency=(time_col, lambda x: (snapshot_date - x.max()).days),
            Frequency=(time_col, "count"),
            Monetary=(amount_col, "sum"),
        )
        .reset_index()
    )

    return rfm


# =========================================================
# 2. RFM SCALING + K-MEANS CLUSTERING
# =========================================================
def cluster_rfm(
    rfm_df: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scale RFM features and perform K-Means clustering.

    Returns:
    - rfm_df with cluster labels
    - cluster centers (original scale)
    """

    features = ["Recency", "Frequency", "Monetary"]
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[features])

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )

    rfm_df["cluster"] = kmeans.fit_predict(rfm_scaled)

    # Inverse-transform cluster centers for interpretability
    centers_scaled = kmeans.cluster_centers_
    centers = pd.DataFrame(
        scaler.inverse_transform(centers_scaled),
        columns=features
    )
    centers["cluster"] = centers.index

    return rfm_df, centers


# =========================================================
# 3. HIGH-RISK CLUSTER IDENTIFICATION
# =========================================================
def assign_high_risk_label(
    rfm_df: pd.DataFrame,
    cluster_centers: pd.DataFrame
) -> pd.DataFrame:
    """
    Identify high-risk cluster and assign is_high_risk label.

    High-risk definition:
    - High Recency (inactive customers)
    - Low Frequency
    - Low Monetary value
    """

    centers = cluster_centers.copy()

    # Rank clusters: higher score = higher risk
    centers["risk_score"] = (
        centers["Recency"].rank(ascending=False) +
        centers["Frequency"].rank(ascending=True) +
        centers["Monetary"].rank(ascending=True)
    )

    high_risk_cluster = centers.sort_values(
        "risk_score", ascending=False
    ).iloc[0]["cluster"]

    rfm_df["is_high_risk"] = (
        rfm_df["cluster"] == high_risk_cluster
    ).astype(int)

    return rfm_df


# =========================================================
# 4. MERGE WITH FEATURE TABLE
# =========================================================
def merge_with_features(
    feature_df: pd.DataFrame,
    rfm_labeled_df: pd.DataFrame,
    customer_id_col: str = "CustomerId"
) -> pd.DataFrame:
    """
    Merge proxy target with engineered features.
    """

    merged = feature_df.merge(
        rfm_labeled_df[[customer_id_col, "is_high_risk"]],
        on=customer_id_col,
        how="left"
    )

    return merged


# =========================================================
# 5. FULL PIPELINE (TASK 4 ENTRY POINT)
# =========================================================
def proxy_target_pipeline(
    transactions: pd.DataFrame,
    feature_df: pd.DataFrame,
    customer_id_col: str = "CustomerId",
    amount_col: str = "Amount",
    time_col: str = "TransactionStartTime",
    snapshot_date: pd.Timestamp | None = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full proxy target engineering pipeline.

    Returns:
    - rfm_labeled_df  : CustomerId → RFM + cluster + is_high_risk
    - features_with_target : Modeling-ready dataset
    - cluster_centers : Cluster characteristics table
    """

    # Step 1: RFM
    rfm = calculate_rfm(
        transactions,
        customer_id_col,
        amount_col,
        time_col,
        snapshot_date
    )

    # Step 2: Clustering
    rfm_clustered, centers = cluster_rfm(rfm)

    # Step 3: Risk labeling
    rfm_labeled = assign_high_risk_label(
        rfm_clustered,
        centers
    )

    # Step 4: Merge with Task 3 features
    features_with_target = merge_with_features(
        feature_df,
        rfm_labeled,
        customer_id_col
    )

    # Audit log
    print("Proxy target distribution:")
    print(features_with_target["is_high_risk"].value_counts(normalize=True))

    return rfm_labeled, features_with_target, centers
