"""
ingestion/psp_source.py — PSP (Mollie / Adyen) transaction loader.
"""

import pandas as pd
from pathlib import Path
from ingestion.base import DataSource

REQUIRED_COLUMNS = [
    "psp_transaction_id", "merchant_id", "psp_name", "order_date",
    "settlement_date", "gross_amount", "fee_amount", "net_amount",
    "status", "refund_amount",
]


class PSPTransactionSource(DataSource):
    """Loads PSP transaction data (parquet for MVP, API adapter later)."""

    def __init__(self, data_path: Path = Path("data/psp_transactions.parquet")):
        super().__init__(source_name="psp", data_path=data_path)

    def load(self) -> pd.DataFrame:
        df = pd.read_parquet(self.data_path)
        df["order_date"] = pd.to_datetime(df["order_date"])
        df["settlement_date"] = pd.to_datetime(df["settlement_date"])
        return df

    def validate(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        issues = []
        # Check required columns
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            issues.append(f"Missing columns: {missing}")

        # Check for nulls in critical fields
        for col in ["merchant_id", "order_date", "gross_amount", "status"]:
            null_pct = df[col].isna().mean()
            if null_pct > 0.01:
                issues.append(f"{col} has {null_pct:.1%} nulls")

        # Validate status values
        valid_statuses = {"paid", "refunded", "chargeback", "failed"}
        invalid = set(df["status"].unique()) - valid_statuses
        if invalid:
            issues.append(f"Invalid statuses found: {invalid}")

        # Check date range
        date_range = (df["order_date"].max() - df["order_date"].min()).days
        if date_range < 180:
            issues.append(f"Only {date_range} days of PSP history (need >= 180)")

        return (len(issues) == 0, issues)
