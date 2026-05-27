"""
ingestion/reconciliation.py — Cross-source reconciliation checks.
"""

import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def reconcile_bank_psp(
    bank_df: pd.DataFrame,
    psp_df: pd.DataFrame,
    merchant_id: str,
) -> dict:
    """
    Compare bank PSP settlement credits against PSP net payouts.
    Returns reconciliation result per merchant.

    Business plan: "Bank, PSP, webshop and marketplace records must
    reconcile with each other; mismatches trigger manual review."
    """
    # Bank side: sum of credits categorized as PSP settlements
    bank_psp = bank_df[
        (bank_df["merchant_id"] == merchant_id)
        & (bank_df["category"] == "psp_settlement")
    ]
    bank_total = bank_psp["amount"].sum()

    # PSP side: sum of net settled amounts (paid only)
    psp_settled = psp_df[
        (psp_df["merchant_id"] == merchant_id)
        & (psp_df["status"] == "paid")
    ]
    psp_total = psp_settled["net_amount"].sum()

    # Compute delta
    if psp_total == 0:
        delta = 1.0  # No PSP data -> max mismatch
    else:
        delta = abs(bank_total - psp_total) / psp_total

    return {
        "merchant_id": merchant_id,
        "bank_psp_credits": round(bank_total, 2),
        "psp_net_settled": round(psp_total, 2),
        "recon_delta": round(delta, 4),
        "recon_pass": delta <= config.MAX_RECON_DELTA,
        "flag": "RECON_FAIL" if delta > config.MAX_RECON_DELTA else None,
    }
