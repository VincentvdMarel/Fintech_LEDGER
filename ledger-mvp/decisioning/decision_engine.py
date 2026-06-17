"""
decisioning/decision_engine.py — Assemble final decision envelope.

Combines:
  1. Champion policy result (the actual decision)
  2. Shadow ML score (informational only)
  3. Human-readable explanations
  4. Audit log entry
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import duckdb
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def make_decision(
    merchant_id: str,
    features: dict,
    policy_result: dict,
    ml_score: float | None,
) -> dict:
    """
    Produce the final decision envelope for one application.

    Parameters
    ----------
    merchant_id : str
    features : dict — computed feature vector
    policy_result : dict — output of credit_policy() (the ACTUAL decision)
    ml_score : float | None — shadow model calibrated P(default)

    The shadow ML score is LAYER 4: logged for monitoring only. It is attached
    to the envelope for the audit trail but never alters policy_result.

    Returns
    -------
    dict — full decision envelope (logged and returned to UI)
    """
    # Build explanation narrative
    explanations = policy_result.get("explanations", [])

    # Add ML context (informational / shadow only — never changes the decision)
    if ml_score is not None:
        if ml_score > 0.3:
            ml_note = (
                f"Shadow ML model estimates elevated risk "
                f"(score: {ml_score:.2f}). This is for monitoring only "
                f"and did NOT influence the decision."
            )
        else:
            ml_note = (
                f"Shadow ML model estimates lower risk "
                f"(score: {ml_score:.2f}). Informational only."
            )
    else:
        ml_note = "Shadow ML model not yet available."

    # Assemble envelope
    envelope = {
        "merchant_id": merchant_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decision": policy_result["decision"],
        "max_amount_eur": policy_result["max_amount"],
        "pricing_band": policy_result["pricing_band"],
        "apr": (
            config.PRICING_BANDS.get(policy_result["pricing_band"])
            if policy_result["pricing_band"]
            else None
        ),
        "tenor_max_months": policy_result["tenor_max_months"],
        "reason_codes": policy_result["reason_codes"],
        "explanations": explanations,
        "manual_review_flags": policy_result["manual_review_flags"],
        "signal_pass_rate": policy_result.get("signal_pass_rate"),  # display-only metric
        "shadow_ml_score": round(ml_score, 4) if ml_score is not None else None,
        "shadow_ml_note": ml_note,
        "model_version": "champion_v1.0",
        "requires_credit_officer": True,  # Year 1: always
    }

    # Persist to decision log
    _log_decision(envelope, features)

    return envelope


def _log_decision(envelope: dict, features: dict) -> None:
    """Append decision + features to DuckDB audit log."""
    db_path = config.DUCKDB_PATH
    Path(db_path).parent.mkdir(exist_ok=True)

    con = duckdb.connect(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS decision_log (
            merchant_id     VARCHAR,
            timestamp       VARCHAR,
            decision        VARCHAR,
            max_amount_eur  DOUBLE,
            pricing_band    VARCHAR,
            reason_codes    VARCHAR,
            shadow_ml_score DOUBLE,
            features_json   VARCHAR,
            model_version   VARCHAR
        )
    """)
    con.execute(
        """
        INSERT INTO decision_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            envelope["merchant_id"],
            envelope["timestamp"],
            envelope["decision"],
            envelope["max_amount_eur"],
            envelope["pricing_band"],
            json.dumps(envelope["reason_codes"]),
            envelope["shadow_ml_score"],
            json.dumps({k: (v if not isinstance(v, float) or v == v else None)
                        for k, v in features.items()}),
            envelope["model_version"],
        ],
    )
    con.close()
