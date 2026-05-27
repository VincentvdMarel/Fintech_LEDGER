"""
ingestion/base.py — Abstract data source interface.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataSource(ABC):
    """Base class for all data sources in the Ledger MVP pipeline."""

    def __init__(self, source_name: str, data_path: Path):
        self.source_name = source_name
        self.data_path = data_path
        self._loaded_at: datetime | None = None

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load and return raw data as a DataFrame."""
        ...

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        """Validate schema and data quality. Returns (is_valid, issues)."""
        ...

    def load_and_validate(self) -> pd.DataFrame:
        """Load data, validate, log result, and return."""
        df = self.load()
        self._loaded_at = datetime.utcnow()
        is_valid, issues = self.validate(df)

        if not is_valid:
            logger.warning(
                f"[{self.source_name}] Validation issues: {issues}"
            )
        else:
            logger.info(
                f"[{self.source_name}] Loaded {len(df)} rows, "
                f"validated OK at {self._loaded_at.isoformat()}"
            )
        return df
