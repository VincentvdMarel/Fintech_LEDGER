"""
features/missing_handler.py — Missing data handling rules.

All defaults are PESSIMISTIC (assume worst plausible value).
This is a deliberate design choice for conservative underwriting.

This module is the canonical feature registry — every feature computed in
pipeline.py must have a corresponding entry in PESSIMISTIC_DEFAULTS.

Features NOT yet computable (missing data sources) are listed at the bottom
as stubs so they are tracked and ready to activate when data arrives.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Defaults for features computable from existing data (bank + psp + merchants)
# ---------------------------------------------------------------------------

PESSIMISTIC_DEFAULTS: dict = {

    # --- Group G: Webshop ---
    "cancellation_rate_90d":            0.10,
    "return_rate_90d":                  0.10,
    "order_volume_trend_90d":          -0.10,
    "fulfillment_timeliness_pct":       0.70,
    "repeat_customer_rate_90d":         0.20,
    "aov_stability_cv":                 0.50,
    "sku_concentration_hhi":            1.0,
    "discount_intensity_90d":           0.20,

    # --- Group H: Marketplace ---
    "marketplace_account_health_avg":   0.0,
    "marketplace_account_health_trend": -0.10,
    "marketplace_payout_hold_flag":     True,
    "marketplace_late_shipment_rate":   0.20,
    "marketplace_negative_feedback_rate": 0.10,
    "marketplace_policy_violations_6m": 3,
    "marketplace_gmv_share":            0.0,
    "marketplace_payout_lag_avg":       14.0,

    # --- Group I: Accounting ---
    "gross_margin_avg_6m":              0.0,
    "gross_margin_trend":              -0.10,
    "ebitda_margin_avg":               -0.10,
    "vat_punctuality":                  0.0,
    "ap_days_avg":                      90.0,
    "ar_days_avg":                      60.0,
    "revenue_bank_delta_avg":           0.20,
    "owner_draw_ratio_6m":              0.20,

    # --- Group J: KYC ---
    "ubo_director_match":               False,
    "ubo_verified":                     False,
    "identity_changes_12m":             3,
    "kyc_days_since_last":              730.0,
    "kyc_verification_level_score":     0.33,

    # --- PSP advanced ---
    "dispute_win_rate":                 0.0,
    "buyer_concentration_hhi":          1.0,

    # --- Group A: Cashflow Stability ---
    "monthly_net_cashflow_avg_3m":      0.0,    # no cashflow data
    "monthly_net_cashflow_avg_6m":      0.0,
    "cashflow_trend_delta":             -0.20,  # assume declining
    "revenue_volatility_30d":           0.50,   # high volatility assumed
    "revenue_volatility_90d":           0.50,
    "revenue_volatility_180d":          0.50,
    "revenue_volatility_delta":         0.10,   # assume worsening
    "net_cashflow_coverage":            0.0,    # no coverage
    "negative_balance_pct_90d":         1.0,    # always negative
    "overdraft_dependency":             1.0,    # always overdrawn
    "days_cash_on_hand":                0.0,    # zero buffer
    "inflow_concentration_hhi":         1.0,    # single inflow source (worst)

    # --- Group B: Settlement & Payment Delays ---
    "settlement_delay_p95":             10.0,   # at red threshold
    "settlement_delay_median":          5.0,
    "settlement_delay_std":             5.0,
    "settlement_timing_variability":    1.0,    # highly variable
    "supplier_pay_punctuality":         0.70,   # at red boundary
    "supplier_payout_lumpiness":        1.0,    # very lumpy

    # --- Group C: Refunds & Chargebacks ---
    "refund_rate_30d":                  0.05,   # 5% refund rate
    "refund_rate_90d":                  0.05,
    "refund_rate_ltm":                  0.05,
    "chargeback_rate_30d":              0.01,   # 1%
    "chargeback_rate_90d":              0.01,
    "chargeback_rate_ltm":              0.01,
    "refund_trend_3m":                  0.0,    # no trend data

    # --- Group D: Concentration, Seasonality & GPV ---
    "platform_concentration":           1.0,    # single platform (worst HHI)
    "seasonality_index":                2.0,    # high seasonality
    "gpv_trend_90d":                    -0.10,  # assume slightly declining
    "payment_method_card_pct":          0.25,   # moderate card exposure

    # --- Group E: Operational ---
    "ad_spend_ratio_3m":                0.20,   # moderate assumption

    # --- Group F: Reconciliation & Data Quality ---
    "bank_psp_recon_delta":             1.0,    # max mismatch
    "data_coverage_score":              0.0,    # no data

    # --- Merchant / Application info ---
    "trading_months":                   0.0,
    "monthly_gmv_avg_6m":               0.0,
}


# ---------------------------------------------------------------------------
# Stubs for features requiring future data sources
# Will be activated once the corresponding ingestion module is built.
# ---------------------------------------------------------------------------

FUTURE_DEFAULTS: dict = {
    # Features still requiring data sources not yet generated.
    # Activate by moving to PESSIMISTIC_DEFAULTS once the source is available.

    # PSP advanced (need authorization data from PSP)
    "authorization_capture_ratio":      0.80,
    "fraud_flag_rate_90d":              0.05,

    # KYC deep (need UBO registry API)
    "circular_transaction_flag":        1.0,
    "orders_captures_mismatch":         0.20,
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def apply_defaults(features: pd.Series) -> pd.Series:
    """
    Fill missing/NaN feature values with pessimistic defaults.
    Only applies defaults for features listed in PESSIMISTIC_DEFAULTS.
    Returns a copy with defaults applied.
    """
    filled = features.copy()
    for feat_name, default_val in PESSIMISTIC_DEFAULTS.items():
        if feat_name in filled.index:
            if pd.isna(filled[feat_name]):
                filled[feat_name] = default_val
        # If the feature is absent entirely, add it with the default
        else:
            if feat_name not in ("merchant_id", "kvk_active", "sanctions_hit",
                                  "requested_amount", "requested_tenor_months"):
                filled[feat_name] = default_val
    return filled


def check_data_quality(feature_df: pd.DataFrame) -> dict:
    """
    Check feature-level missing rates across the portfolio.
    Returns dict of features with >40% missing (triggers data quality alert).
    """
    alerts = {}
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        missing_pct = feature_df[col].isna().mean()
        if missing_pct > 0.40:
            alerts[col] = {
                "missing_pct": round(missing_pct, 3),
                "status": "DATA_QUALITY_ALERT",
                "action": (
                    f"Feature '{col}' has {missing_pct:.1%} missing. "
                    f"Investigate data source before using in production."
                ),
            }
    return alerts
