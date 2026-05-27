"""
ingestion/marketplace_source.py — Marketplace data loader (Source D).
Covers bol.com and Amazon monthly performance snapshots.
"""

import pandas as pd
from pathlib import Path
from ingestion.base import DataSource

REQUIRED_COLUMNS = [
    "merchant_id", "marketplace", "month", "gmv", "order_count",
    "cancelled_count", "late_shipment_count", "return_count",
    "negative_feedback_count", "account_health_score",
    "payout_hold_active", "payout_lag_days", "policy_violation_count",
    "marketplace_gmv_share",
]

VALID_MARKETPLACES = {"bol.com", "amazon"}


class MarketplaceSource(DataSource):
    """Loads marketplace monthly performance data."""

    def __init__(self, data_path: Path = Path("data/marketplace_data.parquet")):
        super().__init__(source_name="marketplace", data_path=data_path)

    def load(self) -> pd.DataFrame:
        df = pd.read_parquet(self.data_path)
        df["month"] = pd.to_datetime(df["month"])
        return df

    def validate(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        issues = []

        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            issues.append(f"Missing columns: {missing}")

        for col in ["merchant_id", "month", "gmv", "account_health_score"]:
            if col in df.columns:
                null_pct = df[col].isna().mean()
                if null_pct > 0.01:
                    issues.append(f"{col} has {null_pct:.1%} nulls")

        if "marketplace" in df.columns:
            invalid = set(df["marketplace"].unique()) - VALID_MARKETPLACES
            if invalid:
                issues.append(f"Unknown marketplaces: {invalid}")

        if "account_health_score" in df.columns:
            out_of_range = ((df["account_health_score"] < 1) | (df["account_health_score"] > 10)).mean()
            if out_of_range > 0:
                issues.append(f"{out_of_range:.1%} of health scores outside 1-10 range")

        return (len(issues) == 0, issues)
