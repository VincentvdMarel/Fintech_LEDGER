"""
features/pipeline.py — Compute all underwriting features for a merchant.

Source groups:
  A — Cashflow Stability              (bank)
  B — Settlement & Payment Delays     (bank + psp)
  C — Refunds & Chargebacks           (psp, multi-window)
  D — Concentration, Seasonality, GPV (bank + psp)
  E — Operational / Ad Spend          (bank)
  F — Reconciliation & Data Quality   (cross-source)
  G — Webshop                         (webshop_orders)    [optional]
  H — Marketplace                     (marketplace_data)  [optional]
  I — Accounting / Bookkeeping        (accounting_records)[optional]
  J — KYC / Identity                  (kyc_data)          [optional]

New sources (G–J) are optional: if not passed the pipeline falls back to
pessimistic defaults from missing_handler.py, so run_pipeline.py works with
or without the extended data files.
"""

import pandas as pd
import numpy as np
from datetime import timedelta

from features.missing_handler import apply_defaults


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _hhi(series: pd.Series) -> float:
    total = series.sum()
    if total == 0:
        return 1.0
    shares = series / total
    return float((shares ** 2).sum())


def _cv(series: pd.Series) -> float:
    if len(series) < 2:
        return 1.0
    mean = series.mean()
    if mean == 0:
        return 1.0
    return float(series.std() / abs(mean))


def _slope_normalized(series: pd.Series) -> float:
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
    gross = subset["gross_amount"].sum()
    if gross == 0:
        return 0.05
    return float(subset.loc[subset["status"] == "refunded", "refund_amount"].sum() / gross)


def _chargeback_rate(subset: pd.DataFrame) -> float:
    n = len(subset)
    if n == 0:
        return 0.01
    return float((subset["status"] == "chargeback").sum() / n)


# ---------------------------------------------------------------------------
# Source A/B/D/E — Bank features
# ---------------------------------------------------------------------------

def _features_bank(bank: pd.DataFrame, today: pd.Timestamp) -> dict:
    f: dict = {}
    d30  = today - timedelta(days=30)
    d90  = today - timedelta(days=90)
    d180 = today - timedelta(days=180)
    d365 = today - timedelta(days=365)

    # A1: Monthly net cashflow — 3m and 6m
    for days, label in [(90, "3m"), (180, "6m")]:
        cutoff = today - timedelta(days=days)
        w = bank[bank["date"] >= cutoff]
        if len(w) > 0:
            monthly = w.set_index("date").resample("ME")["amount"].sum()
            f[f"monthly_net_cashflow_avg_{label}"] = float(monthly.mean())
        else:
            f[f"monthly_net_cashflow_avg_{label}"] = np.nan

    # A2: Cashflow trend delta (3m vs 6m, positive = improving)
    c3 = f["monthly_net_cashflow_avg_3m"]
    c6 = f["monthly_net_cashflow_avg_6m"]
    if pd.notna(c3) and pd.notna(c6) and c6 != 0:
        f["cashflow_trend_delta"] = float((c3 - c6) / abs(c6))
    else:
        f["cashflow_trend_delta"] = 0.0

    # A3: Revenue volatility — CV of daily credits, three windows
    for days, label in [(30, "30d"), (90, "90d"), (180, "180d")]:
        cutoff = today - timedelta(days=days)
        credits = bank[(bank["date"] >= cutoff) & (bank["direction"] == "CREDIT")]
        daily = credits.groupby(credits["date"].dt.date)["amount"].sum()
        min_obs = max(days // 3, 10)
        f[f"revenue_volatility_{label}"] = _cv(daily) if len(daily) >= min_obs else 0.50

    f["revenue_volatility_delta"] = float(
        f["revenue_volatility_30d"] - f["revenue_volatility_90d"]
    )

    # A4: Negative balance pct (90d)
    b90 = bank[bank["date"] >= d90]
    if len(b90) > 0:
        daily_bal = b90.groupby(b90["date"].dt.date)["balance_after"].last()
        f["negative_balance_pct_90d"] = float((daily_bal < 0).mean())
    else:
        f["negative_balance_pct_90d"] = 1.0

    # A5: Days cash on hand (90d)
    if len(b90) > 0:
        avg_bal = b90.groupby(b90["date"].dt.date)["balance_after"].last().mean()
        daily_opex = b90[b90["direction"] == "DEBIT"]["amount"].abs().sum() / 90
        f["days_cash_on_hand"] = float(avg_bal / daily_opex) if daily_opex > 0 else 0.0
    else:
        f["days_cash_on_hand"] = 0.0

    # A6: Overdraft dependency (180d)
    b180 = bank[bank["date"] >= d180]
    if len(b180) > 0:
        daily_bal_180 = b180.groupby(b180["date"].dt.date)["balance_after"].last()
        f["overdraft_dependency"] = float((daily_bal_180 < 0).mean())
    else:
        f["overdraft_dependency"] = 1.0

    # A7: Inflow concentration HHI (90d)
    cr90 = bank[(bank["date"] >= d90) & (bank["direction"] == "CREDIT")]
    if len(cr90) > 0:
        f["inflow_concentration_hhi"] = _hhi(
            cr90.groupby("counterparty_name")["amount"].sum()
        )
    else:
        f["inflow_concentration_hhi"] = 1.0

    # B2: Supplier payment regularity + lumpiness
    sup = bank[(bank["category"] == "supplier_payment") & (bank["date"] >= d180)]
    if len(sup) >= 3:
        monthly_presence = sup.resample("ME", on="date").size()
        n_months = max(len(pd.date_range(d180, today, freq="ME")), 1)
        f["supplier_pay_punctuality"]  = float((monthly_presence > 0).sum() / n_months)
        f["supplier_payout_lumpiness"] = _cv(sup["amount"].abs())
    else:
        f["supplier_pay_punctuality"]  = 0.70
        f["supplier_payout_lumpiness"] = 1.0

    # D2: Seasonality index (12m monthly bank credits)
    cr12m = bank[(bank["date"] >= d365) & (bank["direction"] == "CREDIT")]
    monthly_rev = cr12m.groupby(cr12m["date"].dt.to_period("M"))["amount"].sum()
    if len(monthly_rev) >= 6:
        f["seasonality_index"] = float(monthly_rev.max() / monthly_rev.mean())
    else:
        f["seasonality_index"] = 2.0

    # E1: Ad spend ratio (90d)
    b3m = bank[bank["date"] >= d90]
    ad  = b3m[b3m["category"] == "advertising"]["amount"].abs().sum()
    rev = b3m[b3m["direction"] == "CREDIT"]["amount"].sum()
    f["ad_spend_ratio_3m"] = float(ad / rev) if rev > 0 else 0.20

    # Monthly GMV estimate (trailing 6m from bank credits)
    f["monthly_gmv_avg_6m"] = float(monthly_rev.tail(6).mean()) if len(monthly_rev) >= 1 else 0.0

    return f


# ---------------------------------------------------------------------------
# Source B/C/D — PSP features
# ---------------------------------------------------------------------------

def _features_psp(psp: pd.DataFrame, today: pd.Timestamp) -> dict:
    f: dict = {}
    d30  = today - timedelta(days=30)
    d90  = today - timedelta(days=90)
    d365 = today - timedelta(days=365)

    psp_paid = psp[psp["status"] == "paid"]

    # B1: Settlement delay stats
    if len(psp_paid) > 0:
        delays = (psp_paid["settlement_date"] - psp_paid["order_date"]).dt.days
        f["settlement_delay_p95"]         = float(delays.quantile(0.95))
        f["settlement_delay_median"]      = float(delays.median())
        f["settlement_delay_std"]         = float(delays.std()) if len(delays) > 1 else 0.0
        f["settlement_timing_variability"] = _cv(delays)
    else:
        f["settlement_delay_p95"]         = 10.0
        f["settlement_delay_median"]      = 5.0
        f["settlement_delay_std"]         = 5.0
        f["settlement_timing_variability"] = 1.0

    # C1: Refund + chargeback rates — multi-window
    for days, label in [(30, "30d"), (90, "90d"), (365, "ltm")]:
        sub = psp[psp["order_date"] >= today - timedelta(days=days)]
        f[f"refund_rate_{label}"]     = _refund_rate(sub)
        f[f"chargeback_rate_{label}"] = _chargeback_rate(sub)

    # C2: Refund trend (90d vs prior 90d)
    psp_r90  = psp[psp["order_date"] >= d90]
    psp_p90  = psp[
        (psp["order_date"] >= d90 - timedelta(days=90)) & (psp["order_date"] < d90)
    ]
    f["refund_trend_3m"] = float(_refund_rate(psp_r90) - _refund_rate(psp_p90))

    # D1: Platform concentration HHI
    if len(psp_paid) > 0:
        f["platform_concentration"] = _hhi(
            psp_paid.groupby("psp_name")["gross_amount"].sum()
        )
    else:
        f["platform_concentration"] = 1.0

    # D3: GPV trend (weekly slope, 90d)
    p90_paid = psp_paid[psp_paid["order_date"] >= d90]
    if len(p90_paid) >= 10:
        weekly_gpv = p90_paid.resample("W", on="order_date")["gross_amount"].sum()
        f["gpv_trend_90d"] = _slope_normalized(weekly_gpv)
    else:
        f["gpv_trend_90d"] = 0.0

    # D4: Payment method card pct (90d)
    p90_all = psp[psp["order_date"] >= d90]
    if len(p90_all) > 0 and "payment_method" in p90_all.columns:
        f["payment_method_card_pct"] = float(
            (p90_all["payment_method"] == "creditcard").mean()
        )
    else:
        f["payment_method_card_pct"] = 0.25

    # PSP advanced: dispute win rate (if dispute_outcome column present)
    if "dispute_outcome" in psp.columns:
        cb = psp[psp["status"] == "chargeback"]
        resolved = cb[cb["dispute_outcome"].isin(["won", "lost"])]
        if len(resolved) > 0:
            f["dispute_win_rate"] = float((resolved["dispute_outcome"] == "won").mean())
        else:
            f["dispute_win_rate"] = 0.50
    else:
        f["dispute_win_rate"] = 0.50

    # PSP advanced: buyer concentration HHI (if buyer_id column present)
    if "buyer_id" in psp_paid.columns and len(psp_paid) > 0:
        buyer_rev = psp_paid.groupby("buyer_id")["gross_amount"].sum()
        f["buyer_concentration_hhi"] = _hhi(buyer_rev)
    else:
        f["buyer_concentration_hhi"] = 1.0

    return f


# ---------------------------------------------------------------------------
# Source G — Webshop features
# ---------------------------------------------------------------------------

def _features_webshop(ws: pd.DataFrame, today: pd.Timestamp) -> dict:
    f: dict = {}
    d90 = today - timedelta(days=90)
    ws90 = ws[ws["order_date"] >= d90]

    if len(ws90) < 10:
        return f  # insufficient data; missing_handler fills defaults

    # Cancellation and return rates
    f["cancellation_rate_90d"] = float((ws90["status"] == "cancelled").mean())
    f["return_rate_90d"]       = float((ws90["status"] == "returned").mean())

    # Order volume trend (weekly)
    weekly_orders = ws90.resample("W", on="order_date").size()
    f["order_volume_trend_90d"] = _slope_normalized(weekly_orders)

    # Fulfillment timeliness — % of completed orders shipped within 3 days
    completed = ws90[ws90["status"] == "completed"]
    if len(completed) > 0:
        f["fulfillment_timeliness_pct"] = float(
            (completed["fulfillment_days"] <= 3).mean()
        )
    else:
        f["fulfillment_timeliness_pct"] = 0.0

    # Repeat customer rate
    f["repeat_customer_rate_90d"] = float(ws90["is_repeat_customer"].mean())

    # AOV stability — CV of completed order values
    if len(completed) > 1:
        f["aov_stability_cv"] = _cv(completed["gross_amount"])
    else:
        f["aov_stability_cv"] = 0.50

    # SKU concentration (HHI of completed revenue per SKU)
    if len(completed) > 0 and "sku_id" in completed.columns:
        sku_rev = completed.groupby("sku_id")["gross_amount"].sum()
        f["sku_concentration_hhi"] = _hhi(sku_rev)
    else:
        f["sku_concentration_hhi"] = 1.0

    # Discount intensity — discounts as % of gross revenue
    rev = completed["gross_amount"].sum()
    if rev > 0:
        f["discount_intensity_90d"] = float(completed["discount_amount"].sum() / rev)
    else:
        f["discount_intensity_90d"] = 0.20

    return f


# ---------------------------------------------------------------------------
# Source H — Marketplace features
# ---------------------------------------------------------------------------

def _features_marketplace(mp: pd.DataFrame, today: pd.Timestamp) -> dict:
    f: dict = {}
    d180 = today - timedelta(days=180)
    d90  = today - timedelta(days=90)
    mp6m = mp[mp["month"] >= d180]

    if len(mp6m) == 0:
        return f

    mp6m_sorted = mp6m.sort_values("month")

    # Account health average and trend
    f["marketplace_account_health_avg"] = float(mp6m["account_health_score"].mean())
    f["marketplace_account_health_trend"] = _slope_normalized(
        mp6m_sorted["account_health_score"]
    )

    # Payout hold — any hold active in last 3 months
    mp3m = mp[mp["month"] >= d90]
    f["marketplace_payout_hold_flag"] = bool(
        mp3m["payout_hold_active"].any() if len(mp3m) > 0 else True
    )

    # Rates relative to order volume
    total_orders = mp6m["order_count"].sum()
    if total_orders > 0:
        f["marketplace_late_shipment_rate"]     = float(mp6m["late_shipment_count"].sum() / total_orders)
        f["marketplace_negative_feedback_rate"] = float(mp6m["negative_feedback_count"].sum() / total_orders)
    else:
        f["marketplace_late_shipment_rate"]     = 0.20
        f["marketplace_negative_feedback_rate"] = 0.10

    f["marketplace_policy_violations_6m"] = int(mp6m["policy_violation_count"].sum())
    f["marketplace_gmv_share"]            = float(mp6m["marketplace_gmv_share"].mean())
    f["marketplace_payout_lag_avg"]       = float(mp6m["payout_lag_days"].mean())

    return f


# ---------------------------------------------------------------------------
# Source I — Accounting features
# ---------------------------------------------------------------------------

def _features_accounting(acc: pd.DataFrame, today: pd.Timestamp) -> dict:
    f: dict = {}
    d180 = today - timedelta(days=180)
    acc6m = acc[acc["month"] >= d180].sort_values("month")

    if len(acc6m) == 0:
        return f

    # Gross margin
    f["gross_margin_avg_6m"] = float(acc6m["gross_margin"].mean())
    f["gross_margin_trend"]  = _slope_normalized(acc6m["gross_margin"])

    # EBITDA margin
    total_rev = acc6m["gross_revenue"].sum()
    f["ebitda_margin_avg"] = float(acc6m["ebitda"].sum() / total_rev) if total_rev > 0 else 0.0

    # VAT (BTW) punctuality — fraction of quarters paid on time
    btw_due = acc6m[acc6m["btw_due"] > 0]
    if len(btw_due) > 0:
        f["vat_punctuality"] = float(btw_due["btw_paid_on_time"].mean())
    else:
        f["vat_punctuality"] = 1.0  # no BTW due period = not penalised

    # AP / AR days
    f["ap_days_avg"] = float(acc6m["ap_days"].mean())
    f["ar_days_avg"] = float(acc6m["ar_days"].mean())

    # Revenue-bank reconciliation delta (positive = accounting overstates revenue)
    f["revenue_bank_delta_avg"] = float(acc6m["revenue_bank_delta"].mean())

    # Owner draw ratio
    f["owner_draw_ratio_6m"] = (
        float(acc6m["owner_draws"].sum() / total_rev) if total_rev > 0 else 0.0
    )

    return f


# ---------------------------------------------------------------------------
# Source J — KYC features
# ---------------------------------------------------------------------------

def _features_kyc(kyc_df: pd.DataFrame, merchant_id: str, today: pd.Timestamp) -> dict:
    f: dict = {}
    row = kyc_df[kyc_df["merchant_id"] == merchant_id]

    if len(row) == 0:
        return f

    kyc = row.iloc[0]

    f["ubo_director_match"]        = bool(kyc.get("ubo_director_match", False))
    f["ubo_verified"]              = bool(kyc.get("ubo_verified", False))
    f["identity_changes_12m"]      = int(kyc.get("identity_changes_12m", 3))

    last_kyc = pd.Timestamp(kyc.get("last_kyc_date", today - timedelta(days=730)))
    f["kyc_days_since_last"] = float((today - last_kyc).days)

    level_score = {"basic": 0.33, "enhanced": 0.67, "full": 1.0}
    f["kyc_verification_level_score"] = level_score.get(
        str(kyc.get("verification_level", "basic")), 0.33
    )

    return f


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_features(
    merchant_id: str,
    bank_df: pd.DataFrame,
    psp_df: pd.DataFrame,
    merchant_info: pd.Series,
    application: pd.Series,
    webshop_df: pd.DataFrame | None = None,
    marketplace_df: pd.DataFrame | None = None,
    accounting_df: pd.DataFrame | None = None,
    kyc_df: pd.DataFrame | None = None,
) -> pd.Series:
    """
    Compute the full feature vector for one merchant.

    Parameters
    ----------
    merchant_id    : str
    bank_df        : all merchants' bank transactions
    psp_df         : all merchants' PSP transactions
    merchant_info  : row from merchants table
    application    : row from loan applications table
    webshop_df     : webshop_orders.parquet  (optional — Source G)
    marketplace_df : marketplace_data.parquet (optional — Source H)
    accounting_df  : accounting_records.parquet (optional — Source I)
    kyc_df         : kyc_data.parquet          (optional — Source J)

    Returns
    -------
    pd.Series — full feature vector with pessimistic defaults applied
    """
    today = pd.Timestamp.now().normalize()

    # Filter raw data to this merchant
    bank = bank_df[bank_df["merchant_id"] == merchant_id].copy()
    psp  = psp_df[psp_df["merchant_id"] == merchant_id].copy()

    features: dict = {"merchant_id": merchant_id}

    # --- Sources A/B/D/E: bank ---
    features.update(_features_bank(bank, today))

    # --- Sources B/C/D: PSP ---
    features.update(_features_psp(psp, today))

    # --- Cross-source: net cashflow coverage (needs application data) ---
    requested_monthly = (
        application["requested_amount"] / application["requested_tenor_months"]
    )
    cf6m = features.get("monthly_net_cashflow_avg_6m", 0) or 0
    features["net_cashflow_coverage"] = (
        float(cf6m / requested_monthly) if requested_monthly > 0 else 0.0
    )

    # --- Cross-source: bank–PSP reconciliation delta ---
    psp_paid      = psp[psp["status"] == "paid"]
    bank_credits  = bank[bank["category"] == "psp_settlement"]["amount"].sum()
    psp_net_total = psp_paid["net_amount"].sum() if len(psp_paid) > 0 else 0
    features["bank_psp_recon_delta"] = (
        float(abs(bank_credits - psp_net_total) / psp_net_total)
        if psp_net_total > 0 else 1.0
    )

    # --- Source G: webshop (optional) ---
    if webshop_df is not None and len(webshop_df) > 0:
        ws = webshop_df[webshop_df["merchant_id"] == merchant_id]
        features.update(_features_webshop(ws, today))

    # --- Source H: marketplace (optional) ---
    if marketplace_df is not None and len(marketplace_df) > 0:
        mp = marketplace_df[marketplace_df["merchant_id"] == merchant_id]
        features.update(_features_marketplace(mp, today))

    # --- Source I: accounting (optional) ---
    if accounting_df is not None and len(accounting_df) > 0:
        acc = accounting_df[accounting_df["merchant_id"] == merchant_id]
        features.update(_features_accounting(acc, today))

    # --- Source J: KYC (optional) ---
    if kyc_df is not None and len(kyc_df) > 0:
        features.update(_features_kyc(kyc_df, merchant_id, today))

    # --- Data coverage score (reflects how many sources contributed) ---
    coverage = 0
    if len(bank) >= 30:                                  coverage += 1
    if len(psp) >= 30:                                   coverage += 1
    if len(bank) >= 180:                                 coverage += 1
    if len(psp_paid) >= 10:                              coverage += 1
    if webshop_df is not None and len(webshop_df[webshop_df["merchant_id"] == merchant_id]) >= 10:
        coverage += 1
    if marketplace_df is not None and len(marketplace_df[marketplace_df["merchant_id"] == merchant_id]) > 0:
        coverage += 1
    if accounting_df is not None and len(accounting_df[accounting_df["merchant_id"] == merchant_id]) > 0:
        coverage += 1
    if kyc_df is not None and len(kyc_df[kyc_df["merchant_id"] == merchant_id]) > 0:
        coverage += 1
    features["data_coverage_score"] = coverage / 8.0

    # --- Merchant-level info ---
    features["trading_months"] = float(
        (today - pd.Timestamp(merchant_info["incorporation_date"])).days / 30.44
    )
    features["kvk_active"]    = bool(merchant_info["kvk_active"])
    features["sanctions_hit"] = bool(merchant_info["sanctions_hit"])

    # --- Application info ---
    features["requested_amount"]       = float(application["requested_amount"])
    features["requested_tenor_months"] = float(application["requested_tenor_months"])

    # --- Apply pessimistic defaults to any remaining NaNs ---
    return apply_defaults(pd.Series(features))
