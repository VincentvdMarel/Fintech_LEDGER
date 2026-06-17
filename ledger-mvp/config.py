"""
config.py — Central configuration for Ledger MVP.

SINGLE SOURCE OF TRUTH for every threshold, constant, and feature parameter.
No threshold may be hard-coded anywhere else (policy, UI, exports, or the HTML
deck data generator). The display layers import the resolved signal definitions
from `get_dashboard_signals()` so the dashboard can never drift from the policy.

Architecture (four explicitly separated layers — see policy/credit_policy.py):
  1. Hard knockouts      -> automatic DECLINE                 (policy/knockouts.py)
  2. Scored policy gates -> drive the decision (red/amber)    (SCORED_GATES below)
  3. Dashboard signals   -> display / explanation ONLY        (DASHBOARD_SIGNALS)
  4. Shadow-ML features  -> logged, NOT used in decisions     (models/train_shadow.py)
"""

# ---------------------------------------------------------------------------
# Loan product constraints (from business plan)
# ---------------------------------------------------------------------------
LOAN_MIN_EUR = 10_000
LOAN_MAX_EUR = 150_000
LOAN_MAX_EUR_YEAR1 = 25_000          # Year-1 pilot cap
HARD_CAP_EUR = LOAN_MAX_EUR_YEAR1    # Hard €25k cap — always enforced explicitly
TENOR_MIN_MONTHS = 3
TENOR_MAX_MONTHS = 18

# ---------------------------------------------------------------------------
# Pricing bands (APR)
# ---------------------------------------------------------------------------
PRICING_BANDS = {
    "A": 0.110,   # 11.0%
    "B": 0.125,   # 12.5%  (blended base case)
    "C": 0.140,   # 14.0%
}
# Pricing-band assignment thresholds (see policy._assign_pricing_band)
PRICING_A_MAX_AMBER = 1              # A: 0 red, <= 1 amber, coverage > A_MIN_COVERAGE
PRICING_A_MIN_COVERAGE = 2.0
PRICING_C_MAX_COVERAGE = 1.5        # C: >= 1 red OR coverage < this

# ---------------------------------------------------------------------------
# LAYER 1 — Hard knock-out thresholds (any single hit -> DECLINE)
# The 7 knockouts live in policy/knockouts.py and read these constants.
# ---------------------------------------------------------------------------
MIN_TRADING_MONTHS = 24
MIN_MONTHLY_GMV = 25_000              # EUR 0.3M annual / 12
MAX_RECON_DELTA = 0.20                # >20% bank-PSP mismatch -> decline
MAX_NEGATIVE_BALANCE_PCT = 0.50       # >50% of days negative -> decline
# NOTE: refund rate and chargeback rate are deliberately NOT knockouts.
#       They are scored gates only (see SCORED_GATES below).

# ---------------------------------------------------------------------------
# LAYER 2 — Scored gate thresholds (green / amber / red boundaries)
# These DRIVE the decision via red/amber flag counting.
#   standard gate: value >  red  -> red flag   (higher is worse)
#   inverted gate: value <  red  -> red flag   (lower is worse)
# 15 scored gates (+ marketplace payout-hold amber handled in policy).
# ---------------------------------------------------------------------------
SCORED_GATES = {
    # Bank / cashflow
    "revenue_volatility_90d":           {"green": 0.25, "red": 0.45, "type": "standard"},
    "net_cashflow_coverage":            {"green": 1.80, "red": 1.20, "type": "inverted"},
    "supplier_pay_punctuality":         {"green": 0.90, "red": 0.70, "type": "inverted"},
    "ad_spend_ratio_3m":                {"green": 0.15, "red": 0.30, "type": "standard"},
    # PSP — refund & chargeback are SCORED GATES, not knockouts
    "refund_rate_ltm":                  {"green": 0.03, "red": 0.08,  "type": "standard"},
    "chargeback_rate_ltm":              {"green": 0.005,"red": 0.015, "type": "standard"},  # red = 1.5%
    "settlement_delay_p95":             {"green": 5,    "red": 10,    "type": "standard"},
    "platform_concentration":           {"green": 0.40, "red": 0.65,  "type": "standard"},
    # Accounting (Source I)
    "vat_punctuality":                  {"green": 0.90, "red": 0.70, "type": "inverted"},
    "gross_margin_avg_6m":              {"green": 0.35, "red": 0.15, "type": "inverted"},
    "revenue_bank_delta_avg":           {"green": 0.05, "red": 0.15, "type": "standard"},  # not shown prominently
    # Marketplace (Source H)
    "marketplace_account_health_avg":   {"green": 8.0,  "red": 5.0,  "type": "inverted"},
    "marketplace_late_shipment_rate":   {"green": 0.02, "red": 0.08, "type": "standard"},  # not shown prominently
    # Webshop (Source G)
    "cancellation_rate_90d":            {"green": 0.03, "red": 0.08, "type": "standard"},  # 3% / 8%
    "return_rate_90d":                  {"green": 0.05, "red": 0.12, "type": "standard"},  # 5% / 12%
}

# ---------------------------------------------------------------------------
# Policy decision thresholds (red/amber flag counting)
#   any hard knockout                       -> DECLINE
#   red >= RED_DECLINE_THRESHOLD            -> DECLINE
#   red >= RED_MANUAL_THRESHOLD             -> MANUAL_REVIEW
#   (red + amber) >= TOTAL_FLAGS_MANUAL     -> MANUAL_REVIEW
#   otherwise                               -> APPROVE (still credit-officer reviewed, Year 1)
# ---------------------------------------------------------------------------
RED_DECLINE_THRESHOLD = 3             # >= 3 reds -> decline
RED_MANUAL_THRESHOLD = 2              # exactly 2 reds -> manual review
TOTAL_FLAGS_MANUAL_THRESHOLD = 5      # >= 5 (red + amber) -> manual review

# ---------------------------------------------------------------------------
# Amount calculation
#   limit = min(requested,
#               monthly_cashflow * tenor * CASHFLOW_DEBT_SERVICE_RATIO,
#               monthly_GMV * REVENUE_ANCHOR_MULTIPLIER,
#               HARD_CAP_EUR)
# ---------------------------------------------------------------------------
CASHFLOW_DEBT_SERVICE_RATIO = 0.33    # Max 33% of net cashflow to debt service
REVENUE_ANCHOR_MULTIPLIER = 2.5       # Max loan = 2.5x monthly GMV

# ---------------------------------------------------------------------------
# LAYER 4 — Shadow ML (logged for monitoring, NEVER used in a decision)
# ---------------------------------------------------------------------------
SHADOW_MODEL_PATH = "models/shadow_gbm.pkl"
SHADOW_N_ESTIMATORS = 200
SHADOW_MAX_DEPTH = 4
SHADOW_LEARNING_RATE = 0.05
SHADOW_CALIBRATION_METHOD = "sigmoid"  # CalibratedClassifierCV -> calibrated P(default)
SHADOW_CALIBRATION_CV = 5

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DUCKDB_PATH = "data/ledger_mvp.duckdb"


# ===========================================================================
# LAYER 3 — Dashboard signals (DISPLAY / EXPLANATION ONLY — never decide)
# ===========================================================================
# The 18 dashboard signals shown on the scorecard. They OVERLAP the 15 scored
# gates but do not nest:
#   - 13 of the 18 displayed signals are also scored gates (`scored: True`)
#   - 5 displayed signals are display-only (`scored: False`)
#   - 2 scored gates (revenue_bank_delta_avg, marketplace_late_shipment_rate)
#     are NOT shown prominently on the scorecard.
#
# Thresholds for scored signals are pulled from SCORED_GATES (single source of
# truth). Display-only signals carry their own thresholds here because they do
# not participate in the decision.
# ---------------------------------------------------------------------------
DISPLAY_ONLY_THRESHOLDS = {
    "overdraft_dependency":       {"green": 0.10, "red": 0.50, "type": "standard"},
    "days_cash_on_hand":          {"green": 45,   "red": 15,   "type": "inverted"},
    # bank_psp_recon_delta is a Layer-1 knockout (> MAX_RECON_DELTA). Shown here
    # only for context; its green/red mirror the knockout boundary.
    "bank_psp_recon_delta":       {"green": 0.05, "red": MAX_RECON_DELTA, "type": "standard"},
    "fulfillment_timeliness_pct": {"green": 0.90, "red": 0.70, "type": "inverted"},
}

# Order + metadata for the 18 dashboard signals. `scored` flags policy gates.
DASHBOARD_SIGNALS = [
    # name,                            label,                       source,        fmt,     scored
    ("net_cashflow_coverage",          "Cashflow Coverage Ratio",   "Bank",        "ratio", True),
    ("revenue_volatility_90d",         "Revenue Volatility (90d)",  "Bank",        "pct",   True),
    ("overdraft_dependency",           "Overdraft Dependency",      "Bank",        "pct",   False),
    ("days_cash_on_hand",              "Days Cash on Hand",         "Bank",        "days",  False),
    ("refund_rate_ltm",                "Refund Rate (LTM)",         "PSP",         "pct",   True),
    ("chargeback_rate_ltm",            "Chargeback Rate (LTM)",     "PSP",         "pct",   True),
    ("settlement_delay_p95",           "Settlement Delay P95",      "PSP",         "days",  True),
    ("platform_concentration",         "Platform Concentration",    "PSP",         "hhi",   True),
    ("supplier_pay_punctuality",       "Supplier Pay Regularity",   "Bank",        "pct",   True),
    ("ad_spend_ratio_3m",              "Ad Spend Ratio (3m)",       "Bank",        "pct",   True),
    ("vat_punctuality",                "VAT Punctuality",           "Accounting",  "pct",   True),
    ("gross_margin_avg_6m",            "Gross Margin (6m avg)",     "Accounting",  "pct",   True),
    ("bank_psp_recon_delta",           "Bank–PSP Recon Gap",        "Cross-source","pct",   False),
    ("marketplace_account_health_avg", "Marketplace Health Score",  "Marketplace", "score", True),
    ("cancellation_rate_90d",          "Cancellation Rate (90d)",   "Webshop",     "pct",   True),
    ("return_rate_90d",                "Return Rate (90d)",         "Webshop",     "pct",   True),
    ("fulfillment_timeliness_pct",     "Fulfillment On-Time Rate",  "Webshop",     "pct",   False),
    ("ubo_director_match",             "UBO / Director Match",      "KYC",         "bool",  False),
]


def get_dashboard_signals() -> list[dict]:
    """
    Resolve the 18 dashboard signals into fully-specified dicts for the display
    layer (Streamlit + HTML deck export). Thresholds for scored gates come from
    SCORED_GATES; display-only thresholds come from DISPLAY_ONLY_THRESHOLDS.

    Returns a list of dicts with keys:
        name, label, source, type, fmt, scored, green, red
    """
    resolved = []
    for name, label, source, fmt, scored in DASHBOARD_SIGNALS:
        if fmt == "bool":
            green = red = None
            gtype = "bool"
        elif name in SCORED_GATES:
            g = SCORED_GATES[name]
            green, red, gtype = g["green"], g["red"], g["type"]
        else:
            g = DISPLAY_ONLY_THRESHOLDS[name]
            green, red, gtype = g["green"], g["red"], g["type"]
        resolved.append({
            "name": name, "label": label, "source": source,
            "type": gtype, "fmt": fmt, "scored": scored,
            "green": green, "red": red,
        })
    return resolved
