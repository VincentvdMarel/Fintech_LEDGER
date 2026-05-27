"""
synthetic_gen.py — Generate all synthetic data for Ledger MVP.
Run once:  python -m data.synthetic_gen

Generates 8 parquet files covering all 6 data sources from the spec:
  A) bank_transactions.parquet     — PSD2/bank (extended with tax + NSF)
  B) psp_transactions.parquet      — PSP (extended with buyer_id + dispute_outcome)
  C) webshop_orders.parquet        — NEW: webshop order-level data
  D) marketplace_data.parquet      — NEW: monthly marketplace snapshots
  E) accounting_records.parquet    — NEW: monthly P&L + VAT records
  F) kyc_data.parquet              — NEW: per-merchant KYC/identity records
     merchants.parquet             — updated with has_webshop, primary_marketplace
     loan_applications.parquet     — unchanged
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
from pathlib import Path

SEED = 42
N_MERCHANTS = 200
MONTHS_HISTORY = 12
OUTPUT_DIR = Path("data")
BASE_DATE = date.today() - timedelta(days=MONTHS_HISTORY * 30)

# Dutch BTW (VAT) standard rate; reduced rate (food, books) = 9%
BTW_STANDARD = 0.21
BTW_REDUCED = 0.09

# Sector-specific parameters for realistic Dutch e-commerce profiles
SECTOR_PARAMS = {
    "fashion":     {"aov": 65,  "gross_margin": (0.45, 0.65), "btw": BTW_STANDARD},
    "electronics": {"aov": 150, "gross_margin": (0.20, 0.35), "btw": BTW_STANDARD},
    "home":        {"aov": 70,  "gross_margin": (0.40, 0.60), "btw": BTW_STANDARD},
    "food":        {"aov": 40,  "gross_margin": (0.30, 0.50), "btw": BTW_REDUCED},
    "beauty":      {"aov": 45,  "gross_margin": (0.50, 0.70), "btw": BTW_STANDARD},
}


# ===========================================================================
# Existing generators (updated with additive fields)
# ===========================================================================

def generate_merchants(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Create synthetic merchant profiles resembling Dutch e-commerce SMEs."""
    records = []
    for i in range(n):
        annual_gmv = rng.uniform(300_000, 5_000_000)
        trading_months = rng.integers(12, 84)
        n_psps = rng.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
        is_bad = i < int(n * 0.05)  # first 5% are "bad" merchants

        kvk_digits = "".join([str(d) for d in rng.choice(list(range(10)), size=8)])

        # NEW: channel flags — used by webshop + marketplace generators
        has_webshop = (i % 5) != 0                      # 80% have a webshop
        marketplace_roll = rng.random()
        if marketplace_roll < 0.40:
            primary_marketplace = "bol.com"              # ~40% on bol.com
        elif marketplace_roll < 0.55:
            primary_marketplace = "amazon"               # ~15% on Amazon
        else:
            primary_marketplace = None                   # 45% no marketplace

        records.append({
            "merchant_id":         f"m_{i:04d}",
            "company_name":        f"Shop{i:04d} B.V.",
            "kvk_number":          kvk_digits,
            "incorporation_date":  (
                date.today() - timedelta(days=int(trading_months * 30.44))
            ).isoformat(),
            "annual_gmv":          round(annual_gmv, 2),
            "n_psps":              int(n_psps),
            "sector":              rng.choice(["fashion", "electronics", "home", "food", "beauty"]),
            "sanctions_hit":       False,
            "kvk_active":          True if i != 0 else False,   # merchant 0 = inactive KvK
            "is_bad_synthetic":    is_bad,
            # NEW fields
            "has_webshop":         has_webshop,
            "primary_marketplace": primary_marketplace,
        })
    return pd.DataFrame(records)


def generate_bank_transactions(
    merchants: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Simulate 12 months of daily bank transactions per merchant.

    Categories generated:
      - psp_settlement   (CREDIT)  — PSP daily settlements
      - supplier_payment (DEBIT)   — supplier invoices (~30% of days)
      - advertising      (DEBIT)   — Google/Meta spend (~15% of days)
      - tax_payment      (DEBIT)   — quarterly BTW to Belastingdienst  [NEW]
      - nsf_fee          (DEBIT)   — penalty/NSF fees                  [NEW]
    """
    rows = []

    for _, m in merchants.iterrows():
        daily_revenue = m["annual_gmv"] / 365
        is_bad = m["is_bad_synthetic"]
        sector = m.get("sector", "fashion")
        btw_rate = SECTOR_PARAMS.get(sector, SECTOR_PARAMS["fashion"])["btw"]

        # Quarterly BTW payment months (Dutch quarters: pay in month 4,7,10,1)
        btw_months = {4, 7, 10, 1}
        # Track quarterly revenue accumulation for BTW calculation
        quarterly_revenue_acc = 0.0
        current_quarter_month = None

        for day_offset in range(MONTHS_HISTORY * 30):
            txn_date = BASE_DATE + timedelta(days=day_offset)

            # --- PSP settlement (CREDIT) ---
            noise = rng.normal(1.0, 0.3 if not is_bad else 0.6)
            settlement = round(daily_revenue * max(noise, 0.05), 2)
            rows.append({
                "merchant_id":      m["merchant_id"],
                "transaction_id":   f"txn_{m['merchant_id']}_{day_offset}_cr",
                "date":             txn_date.isoformat(),
                "amount":           settlement,
                "direction":        "CREDIT",
                "counterparty_name": rng.choice(["Mollie B.V.", "Adyen N.V."]),
                "category":         "psp_settlement",
                "balance_after":    None,   # computed at end
            })
            quarterly_revenue_acc += settlement

            # --- Supplier payment (DEBIT, ~30% of days) ---
            if rng.random() < 0.3:
                supplier_amt = round(daily_revenue * rng.uniform(1.0, 3.0) * 0.4, 2)
                rows.append({
                    "merchant_id":      m["merchant_id"],
                    "transaction_id":   f"txn_{m['merchant_id']}_{day_offset}_sup",
                    "date":             txn_date.isoformat(),
                    "amount":           -supplier_amt,
                    "direction":        "DEBIT",
                    "counterparty_name": f"Supplier_{rng.integers(1, 20)}",
                    "category":         "supplier_payment",
                    "balance_after":    None,
                })

            # --- Ad spend (DEBIT, ~15% of days) ---
            if rng.random() < 0.15:
                ad_pct = 0.35 if is_bad else 0.12
                ad_amt = round(daily_revenue * rng.uniform(0.5, 2.0) * ad_pct, 2)
                rows.append({
                    "merchant_id":      m["merchant_id"],
                    "transaction_id":   f"txn_{m['merchant_id']}_{day_offset}_ad",
                    "date":             txn_date.isoformat(),
                    "amount":           -ad_amt,
                    "direction":        "DEBIT",
                    "counterparty_name": rng.choice(["Google Ads", "Meta Platforms"]),
                    "category":         "advertising",
                    "balance_after":    None,
                })

            # --- Quarterly BTW / tax payment (DEBIT) [NEW] ---
            # Pay in the 15th of the BTW payment month
            if txn_date.month in btw_months and txn_date.day == 15:
                btw_due = quarterly_revenue_acc * btw_rate

                # Bad merchants: sometimes late (> day 15) or underpay
                # We approximate by reducing amount and adding a late flag
                if is_bad:
                    payment_fraction = rng.uniform(0.60, 1.00)
                    # Late payment: add a penalty fee (see nsf_fee below)
                else:
                    payment_fraction = rng.uniform(0.98, 1.02)

                btw_payment = round(btw_due * payment_fraction, 2)
                if btw_payment > 0:
                    rows.append({
                        "merchant_id":      m["merchant_id"],
                        "transaction_id":   f"txn_{m['merchant_id']}_{day_offset}_btw",
                        "date":             txn_date.isoformat(),
                        "amount":           -btw_payment,
                        "direction":        "DEBIT",
                        "counterparty_name": "Belastingdienst",
                        "category":         "tax_payment",
                        "balance_after":    None,
                    })
                quarterly_revenue_acc = 0.0  # reset quarterly accumulator

            # --- NSF / penalty fees (DEBIT) [NEW] ---
            # Good: ~0.3 events/month  |  Bad: ~3 events/month
            nsf_lambda = (3.0 if is_bad else 0.3) / 30
            if rng.random() < nsf_lambda:
                fee = round(rng.uniform(15.0, 65.0), 2)
                rows.append({
                    "merchant_id":      m["merchant_id"],
                    "transaction_id":   f"txn_{m['merchant_id']}_{day_offset}_nsf",
                    "date":             txn_date.isoformat(),
                    "amount":           -fee,
                    "direction":        "DEBIT",
                    "counterparty_name": rng.choice(["ABN AMRO Bank", "ING Bank N.V.", "Rabobank"]),
                    "category":         "nsf_fee",
                    "balance_after":    None,
                })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["merchant_id", "date"]).reset_index(drop=True)

    # Compute running balance per merchant (starting at EUR 10,000)
    df["balance_after"] = (
        df.groupby("merchant_id")["amount"].cumsum() + 10_000
    )
    return df


def generate_psp_transactions(
    merchants: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Simulate order-level PSP data with refunds and chargebacks.

    New columns vs original:
      - buyer_id        — hashed customer ID (enables buyer concentration)  [NEW]
      - dispute_outcome — 'won'/'lost' for chargebacks, None for others     [NEW]
    """
    rows = []

    for _, m in merchants.iterrows():
        daily_orders = max(1, int(m["annual_gmv"] / 365 / 45))
        is_bad = m["is_bad_synthetic"]
        refund_prob = rng.uniform(0.08, 0.15) if is_bad else rng.uniform(0.01, 0.04)
        chargeback_prob = 0.02 if is_bad else rng.uniform(0.001, 0.005)

        # Dispute win rate: bad merchants lose more disputes
        dispute_win_rate = rng.uniform(0.10, 0.30) if is_bad else rng.uniform(0.40, 0.70)

        # Buyer pool — power-law distribution (some buyers buy frequently)
        repeat_rate = rng.uniform(0.10, 0.20) if is_bad else rng.uniform(0.25, 0.50)
        monthly_orders = daily_orders * 30
        n_unique_buyers = max(20, int(monthly_orders * (1 - repeat_rate)))
        buyer_weights = rng.exponential(1.0, n_unique_buyers)
        buyer_weights /= buyer_weights.sum()

        for day_offset in range(MONTHS_HISTORY * 30):
            order_date = BASE_DATE + timedelta(days=day_offset)
            n_orders = rng.poisson(daily_orders)

            for j in range(n_orders):
                gross = round(rng.lognormal(mean=3.8, sigma=0.5), 2)
                fee = round(gross * 0.015, 2)
                settlement_lag = int(
                    rng.choice([1, 2, 3, 5, 7], p=[0.4, 0.3, 0.15, 0.1, 0.05])
                )

                r = rng.random()
                if r < chargeback_prob:
                    status = "chargeback"
                    refund_amount = gross
                    dispute_outcome = "won" if rng.random() < dispute_win_rate else "lost"
                elif r < chargeback_prob + refund_prob:
                    status = "refunded"
                    refund_amount = gross
                    dispute_outcome = None
                else:
                    status = "paid"
                    refund_amount = 0.0
                    dispute_outcome = None

                # Buyer ID drawn from power-law pool
                buyer_idx = rng.choice(n_unique_buyers, p=buyer_weights)

                rows.append({
                    "psp_transaction_id": f"psp_{m['merchant_id']}_{day_offset}_{j}",
                    "merchant_id":        m["merchant_id"],
                    "psp_name":           rng.choice(["mollie", "adyen"]),
                    "order_date":         order_date.isoformat(),
                    "settlement_date":    (
                        order_date + timedelta(days=settlement_lag)
                    ).isoformat(),
                    "gross_amount":       gross,
                    "fee_amount":         fee,
                    "net_amount":         round(gross - fee, 2),
                    "status":             status,
                    "refund_amount":      round(refund_amount, 2),
                    "payment_method":     rng.choice(
                        ["ideal", "creditcard", "bancontact", "klarna"],
                        p=[0.55, 0.25, 0.10, 0.10],
                    ),
                    # NEW columns
                    "buyer_id":           f"buyer_{m['merchant_id']}_{buyer_idx:05d}",
                    "dispute_outcome":    dispute_outcome,
                })

    df = pd.DataFrame(rows)
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])
    return df


def generate_loan_applications(
    merchants: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate one loan application per merchant. Unchanged from original."""
    records = []
    for _, m in merchants.iterrows():
        records.append({
            "application_id":         f"app_{m['merchant_id']}",
            "merchant_id":            m["merchant_id"],
            "requested_amount":       int(rng.choice(
                [10_000, 15_000, 20_000, 25_000, 40_000, 50_000, 75_000]
            )),
            "requested_tenor_months": int(rng.choice([3, 6, 9, 12, 18])),
            "purpose":                rng.choice(
                ["inventory", "marketing", "supplier_gap", "seasonal_prep"]
            ),
            "application_date":       date.today().isoformat(),
        })
    return pd.DataFrame(records)


# ===========================================================================
# NEW generators — sources C, D, E, F
# ===========================================================================

def generate_webshop_orders(
    merchants: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Source C — Webshop order-level data.

    Only for merchants where has_webshop = True.
    Captures: order volume, cancellations, returns, AOV, SKU mix,
              fulfillment speed, repeat customers, discount intensity.
    """
    rows = []

    for _, m in merchants.iterrows():
        if not m.get("has_webshop", True):
            continue

        is_bad = m["is_bad_synthetic"]
        sector = m.get("sector", "fashion")
        sector_p = SECTOR_PARAMS.get(sector, SECTOR_PARAMS["fashion"])
        aov = sector_p["aov"] * rng.uniform(0.75, 1.30)

        daily_orders = min(50, max(1, int(m["annual_gmv"] / 365 / aov)))

        # Quality metrics — bad merchants have higher cancel/return/discount
        cancel_rate   = rng.uniform(0.06, 0.15) if is_bad else rng.uniform(0.01, 0.04)
        return_rate   = rng.uniform(0.10, 0.20) if is_bad else rng.uniform(0.02, 0.07)
        repeat_rate   = rng.uniform(0.10, 0.20) if is_bad else rng.uniform(0.25, 0.50)
        discount_freq = rng.uniform(0.20, 0.40) if is_bad else rng.uniform(0.05, 0.15)
        discount_depth = rng.uniform(0.10, 0.30) if is_bad else rng.uniform(0.02, 0.10)

        # Fulfillment speed
        if is_bad:
            fulfil_choices = [1, 2, 3, 5, 7, 10, 14]
            fulfil_probs   = [0.10, 0.15, 0.20, 0.25, 0.15, 0.10, 0.05]
        else:
            fulfil_choices = [1, 2, 3, 5, 7]
            fulfil_probs   = [0.50, 0.25, 0.15, 0.07, 0.03]

        # SKU pool — bad merchants have fewer / more concentrated SKUs
        n_skus = int(rng.integers(3, 20)) if is_bad else int(rng.integers(15, 100))

        # Customer pool for repeat-rate simulation
        monthly_orders = daily_orders * 30
        n_unique_customers = max(20, int(monthly_orders * (1 - repeat_rate)))
        cust_weights = rng.exponential(1.0, n_unique_customers)
        cust_weights /= cust_weights.sum()

        for day_offset in range(MONTHS_HISTORY * 30):
            order_date = BASE_DATE + timedelta(days=day_offset)
            n_orders = rng.poisson(daily_orders)

            for j in range(n_orders):
                r = rng.random()
                if r < cancel_rate:
                    status = "cancelled"
                    fulfillment_days = 0
                elif r < cancel_rate + return_rate:
                    status = "returned"
                    fulfillment_days = int(rng.choice(fulfil_choices, p=fulfil_probs))
                else:
                    status = "completed"
                    fulfillment_days = int(rng.choice(fulfil_choices, p=fulfil_probs))

                gross = round(rng.lognormal(mean=np.log(aov), sigma=0.45), 2)
                discount = (
                    round(gross * rng.uniform(discount_depth * 0.5, discount_depth * 1.5), 2)
                    if rng.random() < discount_freq
                    else 0.0
                )

                cust_idx = rng.choice(n_unique_customers, p=cust_weights)
                is_repeat = rng.random() < repeat_rate

                rows.append({
                    "order_id":           f"ws_{m['merchant_id']}_{day_offset}_{j}",
                    "merchant_id":        m["merchant_id"],
                    "order_date":         order_date.isoformat(),
                    "channel":            rng.choice(
                        ["website", "mobile_app", "api"],
                        p=[0.65, 0.30, 0.05],
                    ),
                    "gross_amount":       gross,
                    "discount_amount":    discount,
                    "item_count":         int(rng.integers(1, 5)),
                    "sku_id":             f"sku_{m['merchant_id']}_{rng.integers(1, n_skus + 1):03d}",
                    "status":             status,
                    "fulfillment_days":   fulfillment_days,
                    "is_repeat_customer": is_repeat,
                    "customer_id":        f"cust_{m['merchant_id']}_{cust_idx:05d}",
                })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df.sort_values(["merchant_id", "order_date"]).reset_index(drop=True)


def generate_marketplace_data(
    merchants: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Source D — Monthly marketplace performance snapshots.

    One row per (merchant, marketplace, month) for merchants with a marketplace.
    Captures: GMV, defect rate, account health, payout holds, feedback, violations.
    """
    rows = []
    months = pd.date_range(start=BASE_DATE, periods=MONTHS_HISTORY, freq="MS")

    for _, m in merchants.iterrows():
        marketplace = m.get("primary_marketplace")
        if marketplace is None:
            continue

        is_bad = m["is_bad_synthetic"]
        monthly_gmv_total = m["annual_gmv"] / 12

        # Marketplace share (30–70% of total GMV)
        mp_share = rng.uniform(0.20, 0.45) if not is_bad else rng.uniform(0.40, 0.70)
        monthly_gmv_mp = monthly_gmv_total * mp_share

        # Account health — bad merchants trend downward
        base_health = rng.uniform(4.0, 7.0) if is_bad else rng.uniform(8.0, 10.0)

        # Rates for bad vs good
        cancel_rate_mp  = rng.uniform(0.04, 0.10) if is_bad else rng.uniform(0.005, 0.02)
        late_ship_rate  = rng.uniform(0.05, 0.15) if is_bad else rng.uniform(0.005, 0.02)
        return_rate_mp  = rng.uniform(0.08, 0.18) if is_bad else rng.uniform(0.01, 0.06)
        neg_feed_rate   = rng.uniform(0.03, 0.08) if is_bad else rng.uniform(0.001, 0.01)
        payout_hold_prob = 0.30 if is_bad else 0.03

        payout_lag_base = rng.uniform(7, 14) if is_bad else rng.uniform(2, 7)

        for i, month_start in enumerate(months):
            # GMV with seasonality
            seasonal_factor = 1.0 + 0.3 * np.sin(2 * np.pi * month_start.month / 12)
            monthly_gmv = max(0, round(
                monthly_gmv_mp * seasonal_factor * rng.normal(1.0, 0.10), 2
            ))

            aov_mp = SECTOR_PARAMS.get(m.get("sector", "fashion"), SECTOR_PARAMS["fashion"])["aov"]
            order_count = max(0, int(monthly_gmv / aov_mp))

            # Health deteriorates over time for bad merchants
            trend = -0.05 * i if is_bad else 0.0
            health_score = round(
                min(10.0, max(1.0, base_health + trend + rng.normal(0, 0.3))), 1
            )

            rows.append({
                "merchant_id":              m["merchant_id"],
                "marketplace":              marketplace,
                "month":                    month_start.date().isoformat(),
                "gmv":                      monthly_gmv,
                "order_count":              order_count,
                "cancelled_count":          int(order_count * cancel_rate_mp * rng.uniform(0.7, 1.3)),
                "late_shipment_count":      int(order_count * late_ship_rate * rng.uniform(0.7, 1.3)),
                "return_count":             int(order_count * return_rate_mp * rng.uniform(0.7, 1.3)),
                "negative_feedback_count":  int(order_count * neg_feed_rate * rng.uniform(0.7, 1.3)),
                "account_health_score":     health_score,
                "payout_hold_active":       rng.random() < payout_hold_prob,
                "payout_lag_days":          round(
                    payout_lag_base * rng.uniform(0.8, 1.5), 1
                ),
                "policy_violation_count":   int(rng.poisson(0.5 if is_bad else 0.05)),
                "marketplace_gmv_share":    round(mp_share, 3),
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["month"] = pd.to_datetime(df["month"])
    return df.sort_values(["merchant_id", "month"]).reset_index(drop=True)


def generate_accounting_records(
    merchants: pd.DataFrame,
    bank_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Source E — Monthly accounting / bookkeeping records.

    Revenue is derived from bank credits with a deliberate reconciliation gap:
      - Good merchants: accounting revenue ≈ bank credits (within 3%)
      - Bad merchants:  accounting revenue > bank credits (inflated, +10-30%)

    Captures: revenue, COGS, gross margin, operating expenses, VAT,
              accounts payable/receivable, owner draws.
    """
    rows = []
    months = pd.date_range(start=BASE_DATE, periods=MONTHS_HISTORY, freq="MS")

    for _, m in merchants.iterrows():
        is_bad = m["is_bad_synthetic"]
        sector = m.get("sector", "fashion")
        sector_p = SECTOR_PARAMS.get(sector, SECTOR_PARAMS["fashion"])
        btw_rate = sector_p["btw"]
        gm_lo, gm_hi = sector_p["gross_margin"]
        gross_margin = rng.uniform(gm_lo, gm_hi)

        # Fixed operating cost base (salaries, rent, SaaS) — monthly
        fixed_opex = m["annual_gmv"] * rng.uniform(0.06, 0.12) / 12

        # AP days: bad merchants are slower to pay
        ap_days_base = rng.uniform(45, 90) if is_bad else rng.uniform(15, 35)
        ar_days_base = rng.uniform(15, 45) if is_bad else rng.uniform(5, 20)

        # Owner draws: bad merchants take more out
        owner_draw_rate = rng.uniform(0.05, 0.15) if is_bad else rng.uniform(0.01, 0.05)

        # Revenue inflation factor for bad merchants (accounting > bank)
        revenue_inflation = rng.uniform(1.10, 1.30) if is_bad else rng.uniform(0.97, 1.03)

        # BTW accumulator for quarterly filing
        btw_accumulated = 0.0
        btw_payment_months = {4, 7, 10, 1}

        for month_start in months:
            # Compute actual bank credits for this merchant in this month
            month_end = month_start + pd.DateOffset(months=1)
            m_bank = bank_df[
                (bank_df["merchant_id"] == m["merchant_id"])
                & (bank_df["date"] >= month_start)
                & (bank_df["date"] < month_end)
                & (bank_df["direction"] == "CREDIT")
            ]
            bank_revenue = m_bank["amount"].sum()

            # Accounting revenue = bank revenue × inflation factor
            gross_revenue = round(bank_revenue * revenue_inflation * rng.uniform(0.98, 1.02), 2)
            if gross_revenue <= 0:
                gross_revenue = round(m["annual_gmv"] / 12, 2)

            # Derived P&L
            cogs = round(gross_revenue * (1 - gross_margin), 2)
            gross_profit = round(gross_revenue - cogs, 2)
            variable_opex = round(gross_revenue * rng.uniform(0.03, 0.08), 2)
            total_opex = round(fixed_opex + variable_opex, 2)
            ebitda = round(gross_profit - total_opex, 2)

            # BTW (VAT) — accrues monthly, paid quarterly
            btw_accrued = round(gross_revenue * btw_rate, 2)
            btw_accumulated += btw_accrued

            if month_start.month in btw_payment_months:
                btw_due = round(btw_accumulated, 2)
                # Bad merchants sometimes underpay
                btw_paid = round(
                    btw_due * (rng.uniform(0.60, 1.00) if is_bad else rng.uniform(0.98, 1.02)),
                    2,
                )
                btw_on_time = not is_bad or rng.random() > 0.4
                btw_accumulated = 0.0
            else:
                btw_due = 0.0
                btw_paid = 0.0
                btw_on_time = True

            # Balance sheet proxies
            accounts_receivable = round(gross_revenue * ar_days_base / 30, 2)
            accounts_payable    = round(cogs * ap_days_base / 30, 2)

            # Cash balance from bank (last balance of month)
            m_bank_all = bank_df[
                (bank_df["merchant_id"] == m["merchant_id"])
                & (bank_df["date"] >= month_start)
                & (bank_df["date"] < month_end)
            ]
            cash_balance = (
                m_bank_all.groupby(m_bank_all["date"].dt.date)["balance_after"]
                .last()
                .dropna()
                .iloc[-1]
                if len(m_bank_all) > 0
                else 0.0
            )

            owner_draws = round(max(0, ebitda) * owner_draw_rate, 2)

            rows.append({
                "merchant_id":        m["merchant_id"],
                "month":              month_start.date().isoformat(),
                "gross_revenue":      gross_revenue,
                "cogs":               cogs,
                "gross_profit":       gross_profit,
                "gross_margin":       round(gross_margin, 4),
                "total_opex":         total_opex,
                "ebitda":             ebitda,
                "accounts_receivable": accounts_receivable,
                "accounts_payable":   accounts_payable,
                "ap_days":            round(ap_days_base * rng.uniform(0.8, 1.2), 1),
                "ar_days":            round(ar_days_base * rng.uniform(0.8, 1.2), 1),
                "cash_balance":       round(cash_balance, 2),
                "btw_accrued":        btw_accrued,
                "btw_due":            btw_due,
                "btw_paid":           btw_paid,
                "btw_paid_on_time":   btw_on_time,
                "owner_draws":        owner_draws,
                # Reconciliation signal: how far does accounting revenue deviate from bank?
                "revenue_bank_delta": round(
                    (gross_revenue - bank_revenue) / bank_revenue
                    if bank_revenue > 0 else 0.0,
                    4,
                ),
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["month"] = pd.to_datetime(df["month"])
    return df.sort_values(["merchant_id", "month"]).reset_index(drop=True)


def generate_kyc_data(
    merchants: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Source F — Per-merchant KYC / identity records.

    One row per merchant.
    Bad merchants: UBO mismatches, more identity changes, lower verification level.
    """
    rows = []
    first_names = ["Jan", "Pieter", "Anna", "Maria", "Thomas", "Sophie", "Lars", "Emma"]
    last_names  = ["de Vries", "van den Berg", "Bakker", "Janssen", "Visser", "Meijer"]

    for _, m in merchants.iterrows():
        is_bad = m["is_bad_synthetic"]

        ubo_name = f"{rng.choice(first_names)} {rng.choice(last_names)}"

        # Director name sometimes differs from UBO for bad merchants
        director_match = not is_bad or rng.random() > 0.40

        # Identity changes: bad merchants change details more often
        identity_changes = int(rng.poisson(2.5 if is_bad else 0.2))

        # Verification level
        if is_bad:
            verification_level = rng.choice(["basic", "enhanced"], p=[0.6, 0.4])
        else:
            verification_level = rng.choice(["enhanced", "full"], p=[0.4, 0.6])

        # KYC verification date (bad merchants may have older/lapsed checks)
        days_since_kyc = int(rng.integers(300, 730)) if is_bad else int(rng.integers(30, 365))
        kyc_date = (date.today() - timedelta(days=days_since_kyc)).isoformat()

        rows.append({
            "merchant_id":          m["merchant_id"],
            "kvk_number":           m["kvk_number"],
            "kvk_verified":         bool(m["kvk_active"]),
            "ubo_name":             ubo_name,
            "ubo_verified":         not is_bad or rng.random() > 0.30,
            "ubo_director_match":   director_match,
            "pep_match":            False,          # simplified: PEP rare
            "sanctions_match":      bool(m["sanctions_hit"]),
            "identity_changes_12m": identity_changes,
            "last_kyc_date":        kyc_date,
            "verification_level":   verification_level,
            "data_source":          "kvk_api",
        })

    return pd.DataFrame(rows)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("Generating merchants...")
    merchants = generate_merchants(N_MERCHANTS, rng)
    merchants.to_parquet(OUTPUT_DIR / "merchants.parquet", index=False)
    print(f"  -> {len(merchants)} merchants saved")

    print("Generating bank transactions (PSP settlements + supplier + ads + tax + NSF)...")
    bank = generate_bank_transactions(merchants, rng)
    bank.to_parquet(OUTPUT_DIR / "bank_transactions.parquet", index=False)
    print(f"  -> {len(bank):,} bank transactions saved")

    print("Generating PSP transactions (with buyer_id + dispute_outcome)...")
    psp = generate_psp_transactions(merchants, rng)
    psp.to_parquet(OUTPUT_DIR / "psp_transactions.parquet", index=False)
    print(f"  -> {len(psp):,} PSP transactions saved")

    print("Generating loan applications...")
    apps = generate_loan_applications(merchants, rng)
    apps.to_parquet(OUTPUT_DIR / "loan_applications.parquet", index=False)
    print(f"  -> {len(apps)} applications saved")

    print("Generating webshop orders (Source C)...")
    webshop = generate_webshop_orders(merchants, rng)
    webshop.to_parquet(OUTPUT_DIR / "webshop_orders.parquet", index=False)
    print(f"  -> {len(webshop):,} webshop orders saved")

    print("Generating marketplace data (Source D)...")
    marketplace = generate_marketplace_data(merchants, rng)
    marketplace.to_parquet(OUTPUT_DIR / "marketplace_data.parquet", index=False)
    print(f"  -> {len(marketplace):,} marketplace monthly snapshots saved")

    print("Generating accounting records (Source E)...")
    accounting = generate_accounting_records(merchants, bank, rng)
    accounting.to_parquet(OUTPUT_DIR / "accounting_records.parquet", index=False)
    print(f"  -> {len(accounting):,} accounting monthly records saved")

    print("Generating KYC data (Source F)...")
    kyc = generate_kyc_data(merchants, rng)
    kyc.to_parquet(OUTPUT_DIR / "kyc_data.parquet", index=False)
    print(f"  -> {len(kyc)} KYC records saved")

    print("\nAll files written to data/")
    print("\nFile summary:")
    for fname in sorted(OUTPUT_DIR.glob("*.parquet")):
        size_kb = fname.stat().st_size / 1024
        print(f"  {fname.name:<40} {size_kb:>8.1f} KB")
