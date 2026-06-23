"""
policy/knockouts.py — LAYER 1: Hard knock-out rules.

This is the SINGLE authoritative implementation of the hard knockouts.
policy/credit_policy.py imports check_knockouts() from here — there is no
second copy of these rules anywhere in the codebase.

Any single knockout triggers an immediate DECLINE, regardless of how clean the
rest of the profile looks (a perfect scorecard is still declined on a sanctions
hit). There are exactly SEVEN knockouts:

    K1  Sanctions / PEP hit
    K2  KvK registration inactive
    K3  UBO / director mismatch
    K4  Bank-to-PSP reconciliation gap > 20%
    K5  Trading history < 24 months
    K6  Monthly GMV < €25,000
    K7  Negative bank balance on > 50% of days

NOTE: refund rate and chargeback rate are NOT knockouts. They are scored gates
(config.SCORED_GATES) and only contribute red/amber flags to the decision.
"""

import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


def check_knockouts(features: dict) -> list[str]:
    """
    Apply the 7 hard knock-out rules.
    Returns the list of triggered reason codes (empty list = all passed).
    """
    codes = []

    # K1: Sanctions / PEP hit
    if features.get("sanctions_hit", False):
        codes.append("SANCTIONS_HIT")

    # K2: KvK registration not active
    if not features.get("kvk_active", False):
        codes.append("KVK_INVALID")

    # K3: UBO / director identity mismatch (KYC) — hard decline per spec §7
    if not features.get("ubo_director_match", False):  # default False = pessimistic
        codes.append("UBO_MISMATCH")

    # K4: Bank-PSP reconciliation gap > 20%
    if features.get("bank_psp_recon_delta", 1.0) > config.MAX_RECON_DELTA:
        codes.append("RECON_FAIL")

    # K5: Trading history < 24 months
    if features.get("trading_months", 0) < config.MIN_TRADING_MONTHS:
        codes.append("INSUFFICIENT_HISTORY")

    # K6: Monthly GMV below floor
    if features.get("monthly_gmv_avg_6m", 0) < config.MIN_MONTHLY_GMV:
        codes.append("REVENUE_TOO_LOW")

    # K7: Negative bank balance on > 50% of the last 90 days
    if features.get("negative_balance_pct_90d", 1.0) > config.MAX_NEGATIVE_BALANCE_PCT:
        codes.append("PERSISTENT_NEGATIVE_BALANCE")

    if codes:
        logger.warning("knockout triggered: %s", codes)
    return codes
