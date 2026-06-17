"""
policy/credit_policy.py — Champion RULES-BASED credit policy for Ledger.

This is a hard-knockout + flag-counting + reason-code engine. A numeric score
NEVER drives any credit decision; the decision is derived exclusively from the
policy rules below. (A display-only "Signal Pass Rate" is computed for the
dashboard but is explicitly not consulted here.)

Four explicitly separated layers (see config.py):
  LAYER 1  Hard knockouts        -> automatic DECLINE        (policy/knockouts.py)
  LAYER 2  Scored policy gates    -> drive decision (red/amber + reason codes)
  LAYER 3  Dashboard signals      -> display only (NOT here — see config/UI)
  LAYER 4  Shadow-ML features      -> logged only (NOT here — see decision_engine)

Decision policy (config thresholds):
  any hard knockout                          -> DECLINE
  red >= RED_DECLINE_THRESHOLD (3)           -> DECLINE
  red >= RED_MANUAL_THRESHOLD  (2)           -> MANUAL_REVIEW
  (red + amber) >= TOTAL_FLAGS_MANUAL (5)    -> MANUAL_REVIEW
  otherwise                                  -> APPROVE (still credit-officer reviewed in Year 1)

Every loan in Year 1 also goes to manual credit-officer review.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from policy.knockouts import check_knockouts          # LAYER 1 — single source
from policy.reason_codes import REASON_EXPLANATIONS


# Reason codes attached to each scored gate. The gate direction (standard vs
# inverted) and its thresholds come from config.SCORED_GATES — the single
# source of truth — so this map only needs feature -> reason code.
GATE_REASON_CODES = {
    "revenue_volatility_90d":         "HIGH_REV_VOLATILITY",
    "refund_rate_ltm":                "HIGH_REFUND_RATE",
    "settlement_delay_p95":           "SETTLEMENT_DELAYS",
    "chargeback_rate_ltm":            "HIGH_CHARGEBACKS",
    "platform_concentration":         "PLATFORM_CONCENTRATION",
    "ad_spend_ratio_3m":              "HIGH_AD_DEPENDENCY",
    "revenue_bank_delta_avg":         "REVENUE_INFLATION_SIGNAL",
    "marketplace_late_shipment_rate": "HIGH_LATE_SHIPMENT_RATE",
    "cancellation_rate_90d":          "HIGH_CANCELLATION_RATE",
    "return_rate_90d":                "HIGH_RETURN_RATE",
    "net_cashflow_coverage":          "WEAK_CASHFLOW_COVERAGE",
    "supplier_pay_punctuality":       "LATE_SUPPLIER_PAYMENTS",
    "vat_punctuality":                "LOW_VAT_PUNCTUALITY",
    "gross_margin_avg_6m":            "LOW_GROSS_MARGIN",
    "marketplace_account_health_avg": "POOR_MARKETPLACE_HEALTH",
}


def _score_gates(f: dict) -> tuple[int, int, list[str]]:
    """
    LAYER 2 — evaluate the 15 scored gates from config.SCORED_GATES.

    Missing features fall back to the gate's RED threshold so that withholding
    data can never improve the outcome (pessimistic by construction; the
    feature pipeline also applies pessimistic defaults upstream).

    Returns (red_count, amber_count, triggered_reason_codes).
    """
    red = 0
    amber = 0
    codes = []

    for feat_name, thresholds in config.SCORED_GATES.items():
        code = GATE_REASON_CODES[feat_name]
        # Pessimistic default: missing value treated as the red boundary.
        value = f.get(feat_name, thresholds["red"])

        if thresholds["type"] == "standard":      # higher = worse
            if value > thresholds["red"]:
                red += 1
                codes.append(code)
            elif value > thresholds["green"]:
                amber += 1
        else:                                      # inverted: lower = worse
            if value < thresholds["red"]:
                red += 1
                codes.append(code)
            elif value < thresholds["green"]:
                amber += 1

    # Marketplace payout hold: extra amber flag regardless of threshold.
    if f.get("marketplace_payout_hold_flag", False):
        amber += 1

    return red, amber, codes


def _calculate_max_amount(f: dict) -> tuple[float, list[str]]:
    """
    Loan sizing:
        limit = min(requested,
                    monthly_cashflow * tenor * CASHFLOW_DEBT_SERVICE_RATIO,
                    monthly_GMV * REVENUE_ANCHOR_MULTIPLIER,
                    HARD_CAP_EUR)          # €25k hard cap — always enforced
    Returns (amount, extra_reason_codes).
    """
    codes = []
    requested   = f.get("requested_amount", 0)
    tenor       = f.get("requested_tenor_months", 6)
    monthly_cf  = f.get("monthly_net_cashflow_avg_6m", 0)
    monthly_gmv = f.get("monthly_gmv_avg_6m", 0)

    cf_cap   = monthly_cf * tenor * config.CASHFLOW_DEBT_SERVICE_RATIO
    rev_cap  = monthly_gmv * config.REVENUE_ANCHOR_MULTIPLIER
    hard_cap = config.HARD_CAP_EUR                      # €25k Year-1 cap, explicit

    # The hard €25k cap is always one of the min() terms — it can never be bypassed.
    max_amount = min(requested, cf_cap, rev_cap, hard_cap)
    max_amount = max(max_amount, 0)                     # floor at zero

    # Flag when the €25k hard cap is the binding constraint (common in the pilot).
    if hard_cap <= min(requested, cf_cap, rev_cap):
        codes.append("YEAR1_CAP")

    return round(max_amount, -2), codes                 # round to nearest €100


def _assign_pricing_band(red: int, amber: int, coverage: float) -> str:
    """
    Pricing band from the risk profile (config thresholds):
        A (11.0%)  -> 0 red, <= 1 amber, coverage > 2.0
        C (14.0%)  -> >= 1 red OR coverage < 1.5
        B (12.5%)  -> everything else (non-declined)
    """
    if red == 0 and amber <= config.PRICING_A_MAX_AMBER and coverage > config.PRICING_A_MIN_COVERAGE:
        return "A"
    if red >= 1 or coverage < config.PRICING_C_MAX_COVERAGE:
        return "C"
    return "B"


def signal_pass_rate(red: int, amber: int) -> float:
    """
    DISPLAY-ONLY health index — NOT a decision input.

    Signal Pass Rate = green gates / total gates, where a gate is "green" when
    it raised neither a red nor an amber flag. This is surfaced on the dashboard
    purely as an at-a-glance health number; the decision is made entirely by the
    rules above and ignores this value.
    """
    total = len(config.SCORED_GATES)
    green = max(total - red - amber, 0)
    return round(green / total, 4) if total else 0.0


def credit_policy(features: dict) -> dict:
    """
    Run the full champion credit policy on a feature vector.

    Returns dict with keys: decision, max_amount, pricing_band, tenor_max_months,
    reason_codes, explanations, manual_review_flags, signal_pass_rate (display-only).
    """
    reason_codes = []
    manual_flags = []

    # ---- LAYER 1: Hard knockouts -> automatic DECLINE --------------------
    knockout_codes = check_knockouts(features)
    if knockout_codes:
        return {
            "decision": "DECLINE",
            "max_amount": 0,
            "pricing_band": None,
            "tenor_max_months": 0,
            "reason_codes": knockout_codes,
            "explanations": [REASON_EXPLANATIONS.get(c, c) for c in knockout_codes],
            "manual_review_flags": [],
            "signal_pass_rate": 0.0,   # display-only; not evaluated for knockouts
        }

    # ---- LAYER 2: Scored gates drive the decision ------------------------
    red, amber, gate_codes = _score_gates(features)
    reason_codes.extend(gate_codes)

    # ---- Decision logic (pure rules — no numeric score involved) ---------
    if red >= config.RED_DECLINE_THRESHOLD:
        decision = "DECLINE"
    elif red >= config.RED_MANUAL_THRESHOLD or (red + amber) >= config.TOTAL_FLAGS_MANUAL_THRESHOLD:
        decision = "MANUAL_REVIEW"
        manual_flags.append(f"{red} red flags, {amber} amber flags")
    else:
        decision = "APPROVE"
        # Year 1: ALL approvals are also routed to a credit officer.
        manual_flags.append("YEAR1_ALL_MANUAL_REVIEW")

    # ---- Amount & pricing (only if not declined) -------------------------
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
        "explanations": [REASON_EXPLANATIONS.get(c, c) for c in reason_codes],
        "manual_review_flags": manual_flags,
        "signal_pass_rate": signal_pass_rate(red, amber),  # LAYER 3 display only
    }
