"""
ingestion/bank_source.py — PSD2 / Open Banking transaction loader.
"""

import pandas as pd
from pathlib import Path
from ingestion.base import DataSource

REQUIRED_COLUMNS = [
    "transaction_id", "merchant_id", "date", "amount",
    "direction", "counterparty_name", "category", "balance_after",
]


class BankTransactionSource(DataSource):
    """Loads PSD2 bank transaction data (parquet for MVP, API adapter later)."""

    def __init__(self, data_path: Path = Path("data/bank_transactions.parquet")):
        super().__init__(source_name="psd2_bank", data_path=data_path)

    def load(self) -> pd.DataFrame:
        df = pd.read_parquet(self.data_path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def validate(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        issues = []
        # Check required columns
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            issues.append(f"Missing columns: {missing}")

        # Check for nulls in critical fields
        for col in ["merchant_id", "date", "amount"]:
            null_pct = df[col].isna().mean()
            if null_pct > 0.01:
                issues.append(f"{col} has {null_pct:.1%} nulls")

        # Check date range
        date_range = (df["date"].max() - df["date"].min()).days
        if date_range < 180:
            issues.append(f"Only {date_range} days of history (need >= 180)")

        return (len(issues) == 0, issues)
