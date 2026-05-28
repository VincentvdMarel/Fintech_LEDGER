"""
policy/credit_policy.py — Champion rules-based credit policy for Ledger.

Every loan in Year 1 also goes to manual credit officer review.
This module produces the recommendation + reason codes.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from policy.reason_codes import REASON_EXPLANATIONS


def _check_knockouts(f: dict) -> list[str]:
    """Apply hard knock-out rules. Returns list of triggered reason codes."""
    codes = []

    if f.get("trading_months", 0) < config.MIN_TRADING_MONTHS:
        codes.append("INSUFFICIENT_HISTORY")

    if not f.get("kvk_active", False):
        codes.append("KVK_INVALID")

    if f.get("bank_psp_recon_delta", 1.0) > config.MAX_RECON_DELTA:
        codes.append("RECON_FAIL")

    if f.get("negative_balance_pct_90d", 1.0) > config.MAX_NEGATIVE_BALANCE_PCT:
        codes.append("PERSISTENT_NEGATIVE_BALANCE")

    if f.get("sanctions_hit", False):
        codes.append("SANCTIONS_HIT")

    if f.get("monthly_gmv_avg_6m", 0) < config.MIN_MONTHLY_GMV:
        codes.append("REVENUE_TOO_LOW")

    # KYC: UBO / director mismatch is a hard decline per spec §7
    if not f.get("ubo_director_match", True):
        codes.append("UBO_MISMATCH")

    return codes


def _score_gates(f: dict) -> tuple[int, int, list[str]]:
    """
    Evaluate scored gates S1-S8.
    Returns (red_count, amber_count, triggered_reason_codes).
    """
    red = 0
    amber = 0
    codes = []

    # Standard gates: higher value = worse
    standard_gates = {
        "revenue_volatility_90d":         "HIGH_REV_VOLATILITY",
        "refund_rate_ltm":                "HIGH_REFUND_RATE",
        "settlement_delay_p95":           "SETTLEMENT_DELAYS",
        "chargeback_rate_ltm":            "HIGH_CHARGEBACKS",
        "platform_concentration":         "PLATFORM_CONCENTRATION",
        "ad_spend_ratio_3m":              "HIGH_AD_DEPENDENCY",
        # New
        "revenue_bank_delta_avg":         "REVENUE_INFLATION_SIGNAL",
        "marketplace_late_shipment_rate": "HIGH_LATE_SHIPMENT_RATE",
        "cancellation_rate_90d":          "HIGH_CANCELLATION_RATE",
        "return_rate_90d":                "HIGH_RETURN_RATE",
    }

    for feat_name, code in standard_gates.items():
        if feat_name not in config.SCORED_GATES:
            continue
        value = f.get(feat_name, config.SCORED_GATES[feat_name]["red"])
        thresholds = config.SCORED_GATES[feat_name]
        if value > thresholds["red"]:
            red += 1
            codes.append(code)
        elif value > thresholds["green"]:
            amber += 1

    # Inverted gates: lower value = worse
    inverted_gates = {
        "net_cashflow_coverage":          "WEAK_CASHFLOW_COVERAGE",
        "supplier_pay_punctuality":       "LATE_SUPPLIER_PAYMENTS",
        # New
        "vat_punctuality":                "LOW_VAT_PUNCTUALITY",
        "gross_margin_avg_6m":            "LOW_GROSS_MARGIN",
        "marketplace_account_health_avg": "POOR_MARKETPLACE_HEALTH",
    }

    for feat_name, code in inverted_gates.items():
        if feat_name not in config.SCORED_GATES:
            continue
        value = f.get(feat_name, config.SCORED_GATES[feat_name]["red"])
        thresholds = config.SCORED_GATES[feat_name]
        if value < thresholds["red"]:
            red += 1
            codes.append(code)
        elif value < thresholds["green"]:
            amber += 1

    # Marketplace payout hold: treat as amber flag regardless of threshold
    if f.get("marketplace_payout_hold_flag", False):
        amber += 1

    return red, amber, codes


def _calculate_max_amount(f: dict) -> tuple[float, list[str]]:
    """Calculate maximum approvable loan amount. Returns (amount, extra_codes)."""
    codes = []
    requested = f.get("requested_amount", 0)
    tenor = f.get("requested_tenor_months", 6)

    monthly_cf = f.get("monthly_net_cashflow_avg_6m", 0)
    monthly_gmv = f.get("monthly_gmv_avg_6m", 0)

    # Three-way cap
    cf_cap = monthly_cf * tenor * config.CASHFLOW_DEBT_SERVICE_RATIO
    rev_cap = monthly_gmv * config.REVENUE_ANCHOR_MULTIPLIER
    product_cap = config.LOAN_MAX_EUR_YEAR1  # Year 1 pilot

    max_amount = min(requested, cf_cap, rev_cap, product_cap)
    max_amount = max(max_amount, 0)  # floor at zero

    if max_amount < requested and product_cap == config.LOAN_MAX_EUR_YEAR1:
        codes.append("YEAR1_CAP")

    return round(max_amount, -2), codes  # round to nearest EUR 100


def _assign_pricing_band(red: int, amber: int, coverage: float) -> str:
    """Assign pricing band A/B/C based on risk profile."""
    if red == 0 and amber <= 1 and coverage > 2.0:
        return "A"
    elif red >= 1 or coverage < 1.5:
        return "C"
    else:
        return "B"


def credit_policy(features: dict) -> dict:
    """
    Run the full champion credit policy on a feature vector.

    Parameters
    ----------
    features : dict — output of compute_features(), as dict

    Returns
    -------
    dict with keys: decision, max_amount, pricing_band, tenor_max_months,
                    reason_codes, explanations, manual_review_flags
    """
    reason_codes = []
    manual_flags = []

    # Step 1: Hard knock-outs
    knockout_codes = _check_knockouts(features)
    if knockout_codes:
        return {
            "decision": "DECLINE",
            "max_amount": 0,
            "pricing_band": None,
            "tenor_max_months": 0,
            "reason_codes": knockout_codes,
            "explanations": [REASON_EXPLANATIONS[c] for c in knockout_codes],
            "manual_review_flags": [],
        }

    # Step 2: Scored gates
    red, amber, gate_codes = _score_gates(features)
    reason_codes.extend(gate_codes)

    # Step 3: Decision logic
    if red >= config.DECLINE_RED_THRESHOLD:
        decision = "DECLINE"
    elif red >= 2 or (amber + red) >= (config.MAX_TOTAL_AMBER_RED_FOR_APPROVE + 1):
        decision = "MANUAL_REVIEW"
        manual_flags.append(f"{red} red flags, {amber} amber flags")
    else:
        decision = "APPROVE"

    # Year 1: ALL approvals also flagged for manual review
    if decision == "APPROVE":
        manual_flags.append("YEAR1_ALL_MANUAL_REVIEW")

    # Step 4: Amount & pricing (only if not declined)
    if decision == "DECLINE":
        max_amount = 0
        pricing_band = None
        tenor_max = 0
    else:
        max_amount, amount_codes = _calculate_max_amount(features)
        reason_codes.extend(amount_codes)
        pricing_band = _assign_pricing_band(
            red, amber, features.get("net_cashflow_coverage", 0)
        )
        tenor_max = min(
            features.get("requested_tenor_months", 6),
            config.TENOR_MAX_MONTHS,
        )

    return {
        "decision": decision,
        "max_amount": max_amount,
        "pricing_band": pricing_band,
        "tenor_max_months": tenor_max,
        "reason_codes": reason_codes,
        "explanations": [
            REASON_EXPLANATIONS.get(c, c) for c in reason_codes
        ],
        "manual_review_flags": manual_flags,
    }
