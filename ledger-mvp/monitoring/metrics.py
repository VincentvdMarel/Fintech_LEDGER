"""
monitoring/metrics.py — MVP metrics computation.
"""

import pandas as pd
import duckdb
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def compute_portfolio_metrics(decisions_df: pd.DataFrame) -> dict:
    """
    Compute key portfolio-level metrics from decision data.

    Returns dict with:
    - approval_rate: % approved or sent to manual review
    - decline_rate: % declined
    - decline_reasons: distribution of decline reason codes
    - avg_approved_amount: mean max_amount for non-declined
    - pricing_distribution: count per pricing band
    - avg_shadow_score_by_decision: mean ML score per decision type
    """
    total = len(decisions_df)
    if total == 0:
        return {"error": "No decisions to analyze"}

    approved = decisions_df[decisions_df["decision"].isin(["APPROVE", "MANUAL_REVIEW"])]
    declined = decisions_df[decisions_df["decision"] == "DECLINE"]

    metrics = {
        "total_applications": total,
        "approval_rate": round(len(approved) / total, 3),
        "decline_rate": round(len(declined) / total, 3),
        "avg_approved_amount": round(approved["max_amount_eur"].mean(), 2) if len(approved) > 0 else 0,
        "pricing_distribution": (
            approved["pricing_band"].value_counts().to_dict()
            if len(approved) > 0 else {}
        ),
    }

    # Decline reason distribution
    all_reasons = []
    for codes in declined["reason_codes"]:
        if isinstance(codes, list):
            all_reasons.extend(codes)
    if all_reasons:
        reason_series = pd.Series(all_reasons)
        metrics["decline_reasons"] = reason_series.value_counts().to_dict()
    else:
        metrics["decline_reasons"] = {}

    # Shadow ML score by decision
    if "shadow_ml_score" in decisions_df.columns:
        metrics["avg_shadow_score_by_decision"] = (
            decisions_df.groupby("decision")["shadow_ml_score"]
            .mean()
            .round(4)
            .to_dict()
        )

    return metrics


def load_decision_log() -> pd.DataFrame:
    """Load the decision log from DuckDB."""
    db_path = config.DUCKDB_PATH
    if not Path(db_path).exists():
        return pd.DataFrame()

    con = duckdb.connect(db_path, read_only=True)
    try:
        df = con.execute("SELECT * FROM decision_log").fetchdf()
        return df
    except Exception:
        return pd.DataFrame()
    finally:
        con.close()
