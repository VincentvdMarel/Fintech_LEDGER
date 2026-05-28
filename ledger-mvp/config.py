"""
config.py — Central configuration for Ledger MVP.
All thresholds, constants, and feature parameters in one place.
"""

# -- Loan product constraints (from business plan) --
LOAN_MIN_EUR = 10_000
LOAN_MAX_EUR = 150_000
LOAN_MAX_EUR_YEAR1 = 25_000          # Year-1 pilot cap
TENOR_MIN_MONTHS = 3
TENOR_MAX_MONTHS = 18

# -- Pricing bands (APR) --
PRICING_BANDS = {
    "A": 0.110,   # 11.0%
    "B": 0.125,   # 12.5%  (blended base case)
    "C": 0.140,   # 14.0%
}

# -- Hard knock-out thresholds --
MIN_TRADING_MONTHS = 24
MIN_MONTHLY_GMV = 25_000              # EUR 0.3M annual / 12
MAX_RECON_DELTA = 0.20                # 20% bank-PSP mismatch -> decline
MAX_NEGATIVE_BALANCE_PCT = 0.50       # >50% days negative -> decline

# -- Scored gate thresholds (green / amber / red boundaries) --
# For standard gates: value > red  = red flag  (higher is worse)
# For inverted gates: value < red  = red flag  (lower is worse) — handled in policy
SCORED_GATES = {
    # Existing gates
    "revenue_volatility_90d":           {"green": 0.25, "red": 0.45},
    "refund_rate_ltm":                  {"green": 0.03, "red": 0.08},
    "settlement_delay_p95":             {"green": 5,    "red": 10},
    "chargeback_rate_ltm":              {"green": 0.005,"red": 0.015},
    "platform_concentration":           {"green": 0.40, "red": 0.65},
    "net_cashflow_coverage":            {"green": 1.80, "red": 1.20},  # inverted
    "supplier_pay_punctuality":         {"green": 0.90, "red": 0.70},  # inverted
    "ad_spend_ratio_3m":                {"green": 0.15, "red": 0.30},

    # New gates — accounting (Source I)
    "vat_punctuality":                  {"green": 0.90, "red": 0.70},  # inverted
    "gross_margin_avg_6m":              {"green": 0.35, "red": 0.15},  # inverted
    "revenue_bank_delta_avg":           {"green": 0.05, "red": 0.15},  # higher = worse (fraud signal)

    # New gates — marketplace (Source H)
    "marketplace_account_health_avg":   {"green": 8.0,  "red": 5.0},   # inverted
    "marketplace_late_shipment_rate":   {"green": 0.02, "red": 0.08},

    # New gates — webshop (Source G)
    "cancellation_rate_90d":            {"green": 0.03, "red": 0.08},
    "return_rate_90d":                  {"green": 0.05, "red": 0.12},
}

# -- Policy decision thresholds --
MAX_RED_FOR_APPROVE = 1               # >= 2 reds -> manual review; >= 3 -> decline
MAX_TOTAL_AMBER_RED_FOR_APPROVE = 4
DECLINE_RED_THRESHOLD = 3

# -- Amount calculation --
CASHFLOW_DEBT_SERVICE_RATIO = 0.33    # Max 33% of net cashflow to debt service
REVENUE_ANCHOR_MULTIPLIER = 2.5       # Max loan = 2.5x monthly GMV

# -- Shadow ML --
SHADOW_MODEL_PATH = "models/shadow_gbm.pkl"
SHADOW_N_ESTIMATORS = 200
SHADOW_MAX_DEPTH = 4
SHADOW_LEARNING_RATE = 0.05

# -- Database --
DUCKDB_PATH = "data/ledger_mvp.duckdb"
