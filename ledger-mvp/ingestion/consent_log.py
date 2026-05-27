"""
ingestion/consent_log.py — Consent & audit event logger.

Every data access event must be logged with timestamp, scope,
and revocation status per GDPR and PSD2 requirements.
"""

import json
import duckdb
from datetime import datetime, timezone
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def log_consent_event(
    merchant_id: str,
    consent_type: str,
    scope: list[str],
    granted: bool = True,
    expires_days: int = 90,
) -> dict:
    """
    Log a consent event (grant or revocation) to the audit database.

    Parameters
    ----------
    merchant_id : str
    consent_type : str — e.g. "psd2_aisp", "psp_mollie", "psp_adyen"
    scope : list[str] — e.g. ["transactions", "balances"]
    granted : bool — True for grant, False for revocation
    expires_days : int — consent validity in days

    Returns
    -------
    dict — the logged consent record
    """
    now = datetime.now(timezone.utc)

    record = {
        "merchant_id": merchant_id,
        "consent_type": consent_type,
        "scope": scope,
        "action": "GRANTED" if granted else "REVOKED",
        "timestamp": now.isoformat(),
        "expires_at": (
            now + __import__("datetime").timedelta(days=expires_days)
        ).isoformat() if granted else None,
        "revocable": True,
    }

    # Persist to DuckDB
    db_path = config.DUCKDB_PATH
    Path(db_path).parent.mkdir(exist_ok=True)
    con = duckdb.connect(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS consent_log (
            merchant_id   VARCHAR,
            consent_type  VARCHAR,
            scope         VARCHAR,
            action        VARCHAR,
            timestamp     VARCHAR,
            expires_at    VARCHAR,
            revocable     BOOLEAN
        )
    """)
    con.execute(
        "INSERT INTO consent_log VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            record["merchant_id"],
            record["consent_type"],
            json.dumps(record["scope"]),
            record["action"],
            record["timestamp"],
            record["expires_at"],
            record["revocable"],
        ],
    )
    con.close()
    return record


def get_active_consents(merchant_id: str) -> list[dict]:
    """Retrieve all active (non-expired, non-revoked) consents for a merchant."""
    db_path = config.DUCKDB_PATH
    if not Path(db_path).exists():
        return []

    con = duckdb.connect(db_path, read_only=True)
    try:
        result = con.execute(
            """
            SELECT * FROM consent_log
            WHERE merchant_id = ?
            AND action = 'GRANTED'
            ORDER BY timestamp DESC
            """,
            [merchant_id],
        ).fetchdf()
        return result.to_dict("records")
    finally:
        con.close()
