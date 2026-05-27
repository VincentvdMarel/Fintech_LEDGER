"""
policy/knockouts.py — Hard knock-out rules (H1-H6).
Any single knock-out triggers an immediate DECLINE.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def check_knockouts(features: dict) -> list[str]:
    """
    Apply hard knock-out rules.
    Returns list of triggered reason codes.
    Empty list means all knock-outs passed.
    """
    codes = []

    # H1: Trading history < 24 months
    if features.get("trading_months", 0) < config.MIN_TRADING_MONTHS:
        codes.append("INSUFFICIENT_HISTORY")

    # H2: KvK registration not active
    if not features.get("kvk_active", False):
        codes.append("KVK_INVALID")

    # H3: Bank-PSP reconciliation mismatch > 20%
    if features.get("bank_psp_recon_delta", 1.0) > config.MAX_RECON_DELTA:
        codes.append("RECON_FAIL")

    # H4: Negative bank balance > 50% of last 90 days
    if features.get("negative_balance_pct_90d", 1.0) > config.MAX_NEGATIVE_BALANCE_PCT:
        codes.append("PERSISTENT_NEGATIVE_BALANCE")

    # H5: Sanctions / PEP hit
    if features.get("sanctions_hit", False):
        codes.append("SANCTIONS_HIT")

    # H6: Monthly GMV below floor
    if features.get("monthly_gmv_avg_6m", 0) < config.MIN_MONTHLY_GMV:
        codes.append("REVENUE_TOO_LOW")

    return codes
