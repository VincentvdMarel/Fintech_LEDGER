"""
ingestion/webshop_source.py — Webshop order data loader (Source C).
"""

import pandas as pd
from pathlib import Path
from ingestion.base import DataSource

REQUIRED_COLUMNS = [
    "order_id", "merchant_id", "order_date", "channel",
    "gross_amount", "discount_amount", "item_count", "sku_id",
    "status", "fulfillment_days", "is_repeat_customer", "customer_id",
]

VALID_STATUSES = {"completed", "cancelled", "returned"}


class WebshopOrderSource(DataSource):
    """Loads webshop order data (parquet for MVP, Shopify API adapter later)."""

    def __init__(self, data_path: Path = Path("data/webshop_orders.parquet")):
        super().__init__(source_name="webshop", data_path=data_path)

    def load(self) -> pd.DataFrame:
        df = pd.read_parquet(self.data_path)
        df["order_date"] = pd.to_datetime(df["order_date"])
        return df

    def validate(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        issues = []

        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            issues.append(f"Missing columns: {missing}")

        for col in ["merchant_id", "order_date", "gross_amount", "status"]:
            if col in df.columns:
                null_pct = df[col].isna().mean()
                if null_pct > 0.01:
                    issues.append(f"{col} has {null_pct:.1%} nulls")

        if "status" in df.columns:
            invalid = set(df["status"].unique()) - VALID_STATUSES
            if invalid:
                issues.append(f"Invalid order statuses: {invalid}")

        if "order_date" in df.columns:
            date_range = (df["order_date"].max() - df["order_date"].min()).days
            if date_range < 90:
                issues.append(f"Only {date_range} days of webshop history (need >= 90)")

        return (len(issues) == 0, issues)
