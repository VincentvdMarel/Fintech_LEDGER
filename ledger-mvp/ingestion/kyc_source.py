"""
ingestion/kyc_source.py — KYC / identity data loader (Source F).
One record per merchant covering KvK, UBO, sanctions, identity history.
"""

import pandas as pd
from pathlib import Path
from ingestion.base import DataSource

REQUIRED_COLUMNS = [
    "merchant_id", "kvk_number", "kvk_verified",
    "ubo_name", "ubo_verified", "ubo_director_match",
    "pep_match", "sanctions_match",
    "identity_changes_12m", "last_kyc_date", "verification_level",
]

VALID_VERIFICATION_LEVELS = {"basic", "enhanced", "full"}


class KYCSource(DataSource):
    """Loads KYC / identity records (parquet for MVP, KvK API adapter later)."""

    def __init__(self, data_path: Path = Path("data/kyc_data.parquet")):
        super().__init__(source_name="kyc", data_path=data_path)

    def load(self) -> pd.DataFrame:
        df = pd.read_parquet(self.data_path)
        df["last_kyc_date"] = pd.to_datetime(df["last_kyc_date"])
        return df

    def validate(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        issues = []

        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            issues.append(f"Missing columns: {missing}")

        for col in ["merchant_id", "kvk_number", "ubo_director_match"]:
            if col in df.columns:
                null_pct = df[col].isna().mean()
                if null_pct > 0.01:
                    issues.append(f"{col} has {null_pct:.1%} nulls")

        if "verification_level" in df.columns:
            invalid = set(df["verification_level"].unique()) - VALID_VERIFICATION_LEVELS
            if invalid:
                issues.append(f"Invalid verification levels: {invalid}")

        # Every active merchant must have a KYC record
        if len(df) == 0:
            issues.append("No KYC records found — cannot underwrite without identity verification")

        return (len(issues) == 0, issues)
