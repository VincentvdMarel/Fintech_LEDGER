"""
ingestion/accounting_source.py — Accounting / bookkeeping data loader (Source E).
Monthly P&L, balance sheet proxies, VAT records per merchant.
"""

import pandas as pd
from pathlib import Path
from ingestion.base import DataSource

REQUIRED_COLUMNS = [
    "merchant_id", "month", "gross_revenue", "cogs", "gross_profit",
    "gross_margin", "total_opex", "ebitda",
    "accounts_receivable", "accounts_payable", "ap_days", "ar_days",
    "cash_balance", "btw_accrued", "btw_due", "btw_paid",
    "btw_paid_on_time", "owner_draws", "revenue_bank_delta",
]


class AccountingSource(DataSource):
    """Loads monthly accounting records (parquet for MVP, Exact/AFAS adapter later)."""

    def __init__(self, data_path: Path = Path("data/accounting_records.parquet")):
        super().__init__(source_name="accounting", data_path=data_path)

    def load(self) -> pd.DataFrame:
        df = pd.read_parquet(self.data_path)
        df["month"] = pd.to_datetime(df["month"])
        return df

    def validate(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        issues = []

        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            issues.append(f"Missing columns: {missing}")

        for col in ["merchant_id", "month", "gross_revenue", "gross_margin"]:
            if col in df.columns:
                null_pct = df[col].isna().mean()
                if null_pct > 0.01:
                    issues.append(f"{col} has {null_pct:.1%} nulls")

        if "gross_margin" in df.columns:
            invalid = ((df["gross_margin"] < 0) | (df["gross_margin"] > 1)).mean()
            if invalid > 0.01:
                issues.append(f"{invalid:.1%} of gross_margin values outside 0-1")

        if "month" in df.columns:
            months = df["month"].nunique()
            if months < 3:
                issues.append(f"Only {months} months of accounting history (need >= 3)")

        return (len(issues) == 0, issues)
