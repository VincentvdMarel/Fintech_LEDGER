"""
policy/reason_codes.py — Reason code to human explanation mapping.
"""

REASON_EXPLANATIONS = {
    "INSUFFICIENT_HISTORY":       "Business has less than 24 months of trading history.",
    "KVK_INVALID":                "KvK registration is not active.",
    "RECON_FAIL":                 "Bank and PSP records show >20% revenue mismatch — possible data integrity issue.",
    "PERSISTENT_NEGATIVE_BALANCE":"Bank account was negative on >50% of the last 90 days.",
    "SANCTIONS_HIT":              "Sanctions or PEP screening returned a positive match.",
    "REVENUE_TOO_LOW":            "Average monthly GMV is below EUR 25,000 minimum threshold.",
    "HIGH_REV_VOLATILITY":        "Revenue volatility (CV) exceeds 0.45 over the past 90 days.",
    "HIGH_REFUND_RATE":           "Refund rate exceeds 8% of gross revenue over the last 12 months.",
    "SETTLEMENT_DELAYS":          "95th-percentile settlement delay exceeds 10 days.",
    "HIGH_CHARGEBACKS":           "Chargeback rate exceeds 1.5% of transactions.",
    "PLATFORM_CONCENTRATION":     "Revenue is highly concentrated on a single platform (HHI > 0.65).",
    "WEAK_CASHFLOW_COVERAGE":     "Net monthly cashflow covers less than 1.2x the loan instalment.",
    "LATE_SUPPLIER_PAYMENTS":     "Less than 70% of supplier payments are made on time.",
    "HIGH_AD_DEPENDENCY":         "Advertising spend exceeds 30% of revenue in the last 3 months.",
    "NO_BANK_DATA":               "No bank transaction data available — cannot underwrite.",
    "LIMITED_DATA":               "PSP data unavailable; max loan amount capped at EUR 15,000.",
    "YEAR1_CAP":                  "Year-1 pilot: maximum loan amount capped at EUR 25,000.",
}
