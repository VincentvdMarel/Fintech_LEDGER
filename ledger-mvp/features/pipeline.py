"""
features/pipeline.py — Compute all underwriting features for a merchant.

Groups (data sources available):
  A — Cashflow Stability          (bank)
  B — Settlement & Payment Delays (bank + psp)
  C — Refunds & Chargebacks       (psp, multi-window)
  D — Concentration, Seasonality & GPV Trend (bank + psp)
  E — Operational / Ad Spend      (bank)
  F — Reconciliation & Data Quality (cross-source)

Not yet computable (missing data sources):
  - Webshop features (source C): cancellation, return, AOV, SKU concentration
  - Marketplace features (source D): account health, payout holds, feedback score
  - Accounting features (source E): VAT punctuality, gross margin, AP/AR days
  - KYC deep features (source F): UBO consistency, circular transactions
  - PSP: dispute win-rate, buyer concentration, authorization-to-capture ratio
"""

import pandas as pd
import numpy as np
from datetime import timedelta

from features.missing_handler import apply_defaults


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _hhi(series: pd.Series) -> float:
    """Herfindahl-Hirschman Index — 1.0 = full concentration, 0 = perfectly split."""
    total = series.sum()
    if total == 0:
        return 1.0
    shares = series / total
    return float((shares ** 2).sum())


def _cv(series: pd.Series) -> float:
    """Coefficient of variation (std / |mean|). Returns 1.0 on empty or zero-mean input."""
    if len(series) < 2:
        return 1.0
    mean = series.mean()
    if mean == 0:
        return 1.0
    return float(series.std() / abs(mean))


def _slope_normalized(series: pd.Series) -> float:
    """
    Linear slope of a series, normalized by its mean.
    Positive → upward trend, negative → downward trend.
    Returns 0.0 if the series is too short or mean is zero.
    """
    if len(series) < 3:
        return 0.0
    mean_val = abs(series.mean())
    if mean_val == 0:
        return 0.0
    x = np.arange(len(series), dtype=float)
    try:
        slope = np.polyfit(x, series.values.astype(float), 1)[0]
        return float(slope / mean_val)
    except Exception:
        return 0.0


def _refund_rate(subset: pd.DataFrame) -> float:
    """Refund value / gross value. Returns 0.05 (pessimistic) on empty data."""
    gross = subset["gross_amount"].sum()
    if gross == 0:
        return 0.05
    refunds = subset.loc[subset["status"] == "refunded", "refund_amount"].sum()
    return float(refunds / gross)


def _chargeback_rate(subset: pd.DataFrame) -> float:
    """Chargeback count / total transaction count. Returns 0.01 on empty data."""
    n = len(subset)
    if n == 0:
        return 0.01
    return float((subset["status"] == "chargeback").sum() / n)


# ---------------------------------------------------------------------------
# Main feature computation
# ---------------------------------------------------------------------------

def compute_features(
    merchant_id: str,
    bank_df: pd.DataFrame,
    psp_df: pd.DataFrame,
    merchant_info: pd.Series,
    application: pd.Series,
) -> pd.Series:
    """
    Compute the full feature vector for one merchant.

    Parameters
    ----------
    merchant_id   : str
    bank_df       : pd.DataFrame — bank transactions (all merchants; filtered inside)
    psp_df        : pd.DataFrame — PSP transactions (all merchants; filtered inside)
    merchant_info : pd.Series   — row from merchants table
    application   : pd.Series   — row from loan applications table

    Returns
    -------
    pd.Series with named features (pessimistic defaults applied for NaNs)
    """
    # Filter to this merchant
    bank = bank_df[bank_df["merchant_id"] == merchant_id].copy()
    psp  = psp_df[psp_df["merchant_id"] == merchant_id].copy()

    today = pd.Timestamp.now().normalize()
    d30   = today - timedelta(days=30)
    d90   = today - timedelta(days=90)
    d180  = today - timedelta(days=180)
    d365  = today - timedelta(days=365)

    features: dict = {"merchant_id": merchant_id}

    # ================================================================
    # GROUP A — Cashflow Stability
    # ================================================================

    # A1: Monthly net cashflow — 3-month and 6-month windows
    for window_days, label in [(90, "3m"), (180, "6m")]:
        cutoff  = today - timedelta(days=window_days)
        bank_w  = bank[bank["date"] >= cutoff]
        if len(bank_w) > 0:
            monthly = bank_w.set_index("date").resample("ME")["amount"].sum()
            features[f"monthly_net_cashflow_avg_{label}"] = float(monthly.mean())
        else:
            features[f"monthly_net_cashflow_avg_{label}"] = np.nan

    # A2: Cashflow trend delta — positive means recent 3m is stronger than 6m avg
    cf_3m = features["monthly_net_cashflow_avg_3m"]
    cf_6m = features["monthly_net_cashflow_avg_6m"]
    if pd.notna(cf_3m) and pd.notna(cf_6m) and cf_6m != 0:
        features["cashflow_trend_delta"] = float((cf_3m - cf_6m) / abs(cf_6m))
    else:
        features["cashflow_trend_delta"] = 0.0

    # A3: Revenue volatility — CV of daily credits, three windows
    for window_days, label in [(30, "30d"), (90, "90d"), (180, "180d")]:
        cutoff  = today - timedelta(days=window_days)
        credits = bank[(bank["date"] >= cutoff) & (bank["direction"] == "CREDIT")]
        daily   = credits.groupby(credits["date"].dt.date)["amount"].sum()
        min_obs = max(window_days // 3, 10)
        features[f"revenue_volatility_{label}"] = (
            _cv(daily) if len(daily) >= min_obs else 0.50
        )

    # Volatility trend: positive = short-window more volatile than medium (worsening)
    features["revenue_volatility_delta"] = float(
        features["revenue_volatility_30d"] - features["revenue_volatility_90d"]
    )

    # A4: Net cashflow coverage ratio (6m avg vs monthly instalment)
    requested_monthly = (
        application["requested_amount"] / application["requested_tenor_months"]
    )
    cf_6m_val = features["monthly_net_cashflow_avg_6m"]
    features["net_cashflow_coverage"] = (
        float(cf_6m_val / requested_monthly)
        if pd.notna(cf_6m_val) and requested_monthly > 0
        else 0.0
    )

    # A5: Negative balance percentage (90d)
    bank_90d = bank[bank["date"] >= d90]
    if len(bank_90d) > 0:
        daily_bal = bank_90d.groupby(bank_90d["date"].dt.date)["balance_after"].last()
        features["negative_balance_pct_90d"] = float((daily_bal < 0).mean())
    else:
        features["negative_balance_pct_90d"] = 1.0

    # A6: Overdraft dependency — % of days with negative closing balance (180d)
    # Distinct from negative_balance_pct_90d: longer window, broader signal
    bank_180d = bank[bank["date"] >= d180]
    if len(bank_180d) > 0:
        daily_bal_180 = (
            bank_180d.groupby(bank_180d["date"].dt.date)["balance_after"].last()
        )
        features["overdraft_dependency"] = float((daily_bal_180 < 0).mean())
    else:
        features["overdraft_dependency"] = 1.0

    # A7: Days cash on hand (90d) — avg daily balance / avg daily operating spend
    if len(bank_90d) > 0:
        avg_balance = (
            bank_90d.groupby(bank_90d["date"].dt.date)["balance_after"]
            .last()
            .mean()
        )
        daily_opex = bank_90d[bank_90d["direction"] == "DEBIT"]["amount"].abs().sum() / 90
        features["days_cash_on_hand"] = (
            float(avg_balance / daily_opex) if daily_opex > 0 else 0.0
        )
    else:
        features["days_cash_on_hand"] = 0.0

    # A8: Inflow concentration HHI (90d) — who is sending money in
    credits_90d = bank[(bank["date"] >= d90) & (bank["direction"] == "CREDIT")]
    if len(credits_90d) > 0:
        by_counterparty = credits_90d.groupby("counterparty_name")["amount"].sum()
        features["inflow_concentration_hhi"] = _hhi(by_counterparty)
    else:
        features["inflow_concentration_hhi"] = 1.0

    # ================================================================
    # GROUP B — Settlement & Payment Delays
    # ================================================================

    psp_paid = psp[psp["status"] == "paid"]

    # B1: Settlement delay statistics
    if len(psp_paid) > 0:
        delays = (psp_paid["settlement_date"] - psp_paid["order_date"]).dt.days
        features["settlement_delay_p95"]         = float(delays.quantile(0.95))
        features["settlement_delay_median"]       = float(delays.median())
        features["settlement_delay_std"]          = float(delays.std()) if len(delays) > 1 else 0.0
        features["settlement_timing_variability"] = _cv(delays)
    else:
        features["settlement_delay_p95"]         = 10.0
        features["settlement_delay_median"]      = 5.0
        features["settlement_delay_std"]         = 5.0
        features["settlement_timing_variability"] = 1.0

    # B2: Supplier payment regularity (FIX: replaces always-True day_of_month <= 30)
    # Metric: fraction of calendar months in 6m window that had ≥1 supplier payment.
    # 1.0 = paying every month (good); 0.5 = skipping months (stress signal).
    supplier_txns = bank[
        (bank["category"] == "supplier_payment") & (bank["date"] >= d180)
    ]
    if len(supplier_txns) >= 3:
        months_with_payment = (
            supplier_txns.resample("ME", on="date").size()
        )
        n_total_months = max(
            len(pd.date_range(d180, today, freq="ME")), 1
        )
        features["supplier_pay_punctuality"] = float(
            (months_with_payment > 0).sum() / n_total_months
        )
        # B3: Supplier payout lumpiness — CV of payment amounts (high = irregular)
        features["supplier_payout_lumpiness"] = _cv(supplier_txns["amount"].abs())
    else:
        features["supplier_pay_punctuality"]  = 0.70
        features["supplier_payout_lumpiness"] = 1.0

    # ================================================================
    # GROUP C — Refunds & Chargebacks (multi-window)
    # ================================================================

    # Three windows: 30d, 90d, LTM
    for window_days, label in [(30, "30d"), (90, "90d"), (365, "ltm")]:
        cutoff = today - timedelta(days=window_days)
        subset = psp[psp["order_date"] >= cutoff]
        features[f"refund_rate_{label}"]     = _refund_rate(subset)
        features[f"chargeback_rate_{label}"] = _chargeback_rate(subset)

    # Refund trend: recent 90d rate vs prior 90d rate (positive = worsening)
    psp_recent_90 = psp[psp["order_date"] >= d90]
    psp_prior_90  = psp[
        (psp["order_date"] >= d90 - timedelta(days=90))
        & (psp["order_date"] < d90)
    ]
    features["refund_trend_3m"] = float(
        _refund_rate(psp_recent_90) - _refund_rate(psp_prior_90)
    )

    # ================================================================
    # GROUP D — Concentration, Seasonality & GPV Trend
    # ================================================================

    # D1: Platform concentration HHI (PSP split of settled revenue)
    if len(psp_paid) > 0:
        psp_shares = psp_paid.groupby("psp_name")["gross_amount"].sum()
        features["platform_concentration"] = _hhi(psp_shares)
    else:
        features["platform_concentration"] = 1.0

    # D2: Seasonality index — peak month / average month (12m revenue)
    bank_12m_credits = bank[
        (bank["date"] >= d365) & (bank["direction"] == "CREDIT")
    ]
    monthly_rev = (
        bank_12m_credits
        .groupby(bank_12m_credits["date"].dt.to_period("M"))["amount"]
        .sum()
    )
    if len(monthly_rev) >= 6:
        features["seasonality_index"] = float(
            monthly_rev.max() / monthly_rev.mean()
        )
    else:
        features["seasonality_index"] = 2.0

    # D3: GPV trend — normalized slope of weekly gross payment volume (90d)
    if len(psp_paid) > 0:
        psp_90d_paid = psp_paid[psp_paid["order_date"] >= d90]
        if len(psp_90d_paid) >= 10:
            weekly_gpv = psp_90d_paid.resample(
                "W", on="order_date"
            )["gross_amount"].sum()
            features["gpv_trend_90d"] = _slope_normalized(weekly_gpv)
        else:
            features["gpv_trend_90d"] = 0.0
    else:
        features["gpv_trend_90d"] = 0.0

    # D4: Payment method mix — % credit card transactions (higher = more chargeback risk)
    psp_90d_all = psp[psp["order_date"] >= d90]
    if len(psp_90d_all) > 0 and "payment_method" in psp_90d_all.columns:
        features["payment_method_card_pct"] = float(
            (psp_90d_all["payment_method"] == "creditcard").mean()
        )
    else:
        features["payment_method_card_pct"] = 0.25

    # ================================================================
    # GROUP E — Operational / Ad Spend
    # ================================================================

    bank_3m = bank[bank["date"] >= d90]
    ad_spend   = bank_3m[bank_3m["category"] == "advertising"]["amount"].abs().sum()
    revenue_3m = bank_3m[bank_3m["direction"] == "CREDIT"]["amount"].sum()
    features["ad_spend_ratio_3m"] = (
        float(ad_spend / revenue_3m) if revenue_3m > 0 else 0.20
    )

    # ================================================================
    # GROUP F — Reconciliation & Data Quality
    # ================================================================

    # F1: Bank–PSP reconciliation delta
    bank_psp_credits = bank[bank["category"] == "psp_settlement"]["amount"].sum()
    psp_net_total    = psp_paid["net_amount"].sum() if len(psp_paid) > 0 else 0
    if psp_net_total > 0:
        features["bank_psp_recon_delta"] = float(
            abs(bank_psp_credits - psp_net_total) / psp_net_total
        )
    else:
        features["bank_psp_recon_delta"] = 1.0

    # F2: Data coverage score — 0.0 to 1.0, higher = more complete data
    coverage = 0
    if len(bank) >= 30:                       coverage += 1   # has bank history
    if len(psp) >= 30:                        coverage += 1   # has PSP history
    if len(bank) >= 180:                      coverage += 1   # sufficient bank depth
    if len(psp_paid) >= 10:                   coverage += 1   # sufficient paid txns
    if pd.notna(merchant_info.get("kvk_number", np.nan)):  coverage += 1
    features["data_coverage_score"] = coverage / 5.0

    # ================================================================
    # Merchant-level info
    # ================================================================
    features["trading_months"] = float(
        (today - pd.Timestamp(merchant_info["incorporation_date"])).days / 30.44
    )
    features["kvk_active"]    = bool(merchant_info["kvk_active"])
    features["sanctions_hit"] = bool(merchant_info["sanctions_hit"])

    # Monthly GMV estimate (trailing 6m from bank credits)
    features["monthly_gmv_avg_6m"] = (
        float(monthly_rev.tail(6).mean()) if len(monthly_rev) >= 1 else 0.0
    )

    # Application info
    features["requested_amount"]       = float(application["requested_amount"])
    features["requested_tenor_months"] = float(application["requested_tenor_months"])

    # ================================================================
    # Apply pessimistic defaults to any remaining NaNs
    # ================================================================
    return apply_defaults(pd.Series(features))
