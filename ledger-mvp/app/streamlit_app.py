"""
app/streamlit_app.py — Ledger MVP Demo UI.
Run:  streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.bank_source import BankTransactionSource
from ingestion.psp_source import PSPTransactionSource
from ingestion.webshop_source import WebshopOrderSource
from ingestion.marketplace_source import MarketplaceSource
from ingestion.accounting_source import AccountingSource
from ingestion.kyc_source import KYCSource
from features.pipeline import compute_features
from policy.credit_policy import credit_policy
from models.train_shadow import score_merchant
from decisioning.decision_engine import make_decision

st.set_page_config(page_title="Ledger MVP", page_icon="📊", layout="wide")
st.title("📊 Ledger MVP — Credit Decision Demo")

PRICING_LABELS = {"A": "11.0%", "B": "12.5%", "C": "14.0%"}

# ---------------------------------------------------------------------------
# Scorecard configuration — 15 key underwriting features
# ---------------------------------------------------------------------------
# type "standard" : higher value = worse risk
# type "inverted" : lower  value = worse risk
# type "bool"     : True = good, False = bad (knockout)
# fmt             : controls value display formatting

KEY_FEATURES = [
    # Cashflow
    {"name": "net_cashflow_coverage",          "label": "Cashflow Coverage Ratio",    "source": "Bank",         "type": "inverted", "green": 1.80,  "red": 1.20,  "fmt": "ratio"},
    {"name": "revenue_volatility_90d",         "label": "Revenue Volatility (90d)",   "source": "Bank",         "type": "standard", "green": 0.25,  "red": 0.45,  "fmt": "pct"},
    {"name": "overdraft_dependency",           "label": "Overdraft Dependency (180d)","source": "Bank",         "type": "standard", "green": 0.10,  "red": 0.50,  "fmt": "pct"},
    {"name": "days_cash_on_hand",              "label": "Days Cash on Hand",          "source": "Bank",         "type": "inverted", "green": 45,    "red": 15,    "fmt": "days"},
    # PSP
    {"name": "refund_rate_ltm",                "label": "Refund Rate (LTM)",          "source": "PSP",          "type": "standard", "green": 0.03,  "red": 0.08,  "fmt": "pct"},
    {"name": "chargeback_rate_ltm",            "label": "Chargeback Rate (LTM)",      "source": "PSP",          "type": "standard", "green": 0.005, "red": 0.015, "fmt": "pct"},
    {"name": "settlement_delay_p95",           "label": "Settlement Delay P95",       "source": "PSP",          "type": "standard", "green": 5,     "red": 10,    "fmt": "days"},
    {"name": "platform_concentration",         "label": "Platform Concentration (HHI)","source": "PSP",         "type": "standard", "green": 0.40,  "red": 0.65,  "fmt": "hhi"},
    # Payments
    {"name": "supplier_pay_punctuality",       "label": "Supplier Pay Regularity",    "source": "Bank",         "type": "inverted", "green": 0.90,  "red": 0.70,  "fmt": "pct"},
    {"name": "ad_spend_ratio_3m",              "label": "Ad Spend Ratio (3m)",        "source": "Bank",         "type": "standard", "green": 0.15,  "red": 0.30,  "fmt": "pct"},
    # Accounting
    {"name": "vat_punctuality",                "label": "VAT Punctuality",            "source": "Accounting",   "type": "inverted", "green": 0.90,  "red": 0.70,  "fmt": "pct"},
    {"name": "gross_margin_avg_6m",            "label": "Gross Margin (6m avg)",      "source": "Accounting",   "type": "inverted", "green": 0.35,  "red": 0.15,  "fmt": "pct"},
    # Reconciliation & marketplace
    {"name": "bank_psp_recon_delta",           "label": "Bank–PSP Recon Gap",         "source": "Cross-source", "type": "standard", "green": 0.05,  "red": 0.20,  "fmt": "pct"},
    {"name": "marketplace_account_health_avg", "label": "Marketplace Health Score",   "source": "Marketplace",  "type": "inverted", "green": 8.0,   "red": 5.0,   "fmt": "score"},
    # Webshop
    {"name": "cancellation_rate_90d",          "label": "Cancellation Rate (90d)",    "source": "Webshop",      "type": "standard", "green": 0.05,  "red": 0.12,  "fmt": "pct"},
    {"name": "return_rate_90d",                "label": "Return Rate (90d)",          "source": "Webshop",      "type": "standard", "green": 0.08,  "red": 0.15,  "fmt": "pct"},
    {"name": "fulfillment_timeliness_pct",     "label": "Fulfillment On-Time Rate",   "source": "Webshop",      "type": "inverted", "green": 0.90,  "red": 0.70,  "fmt": "pct"},
    # KYC
    {"name": "ubo_director_match",             "label": "UBO / Director Match",       "source": "KYC",          "type": "bool",     "green": None,  "red": None,  "fmt": "bool"},
]


def _fmt(value, fmt_type: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if fmt_type == "pct":
        return f"{float(value) * 100:.1f}%"
    if fmt_type == "ratio":
        return f"{float(value):.2f}×"
    if fmt_type == "days":
        return f"{float(value):.1f} days"
    if fmt_type == "hhi":
        return f"{float(value):.3f}"
    if fmt_type == "score":
        return f"{float(value):.1f} / 10"
    if fmt_type == "bool":
        return "✓ Match" if value else "✗ Mismatch"
    return str(round(float(value), 3))


def _threshold_desc(cfg: dict) -> str:
    if cfg["type"] == "bool":
        return "Must match"

    g, r, fmt = cfg["green"], cfg["red"], cfg["fmt"]

    def fv(v):
        if fmt == "pct":   return f"{v*100:.0f}%"
        if fmt == "ratio": return f"{v:.1f}×"
        if fmt == "days":  return f"{v:.0f}d"
        if fmt == "score": return f"{v:.0f}/10"
        return str(v)

    if cfg["type"] == "standard":
        return f"Green ≤ {fv(g)}  |  Red > {fv(r)}"
    else:
        return f"Green ≥ {fv(g)}  |  Red < {fv(r)}"


def _status(cfg: dict, value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "⚪ No data"

    if cfg["type"] == "bool":
        return "🟢 Pass" if value else "🔴 Knockout"

    g, r = cfg["green"], cfg["red"]

    if cfg["type"] == "standard":
        if value <= g:   return "🟢 Pass"
        if value <= r:   return "🟡 Caution"
        return "🔴 Flag"
    else:  # inverted
        if value >= g:   return "🟢 Pass"
        if value >= r:   return "🟡 Caution"
        return "🔴 Flag"


def build_scorecard(feat_dict: dict) -> pd.DataFrame:
    rows = []
    for cfg in KEY_FEATURES:
        value = feat_dict.get(cfg["name"])
        rows.append({
            "Feature":    cfg["label"],
            "Source":     cfg["source"],
            "Value":      _fmt(value, cfg["fmt"]),
            "Status":     _status(cfg, value),
            "Thresholds": _threshold_desc(cfg),
        })
    return pd.DataFrame(rows)


def _style_status(val: str) -> str:
    if "🟢" in val:
        return "background-color: #d4edda; color: #155724; font-weight: bold"
    if "🟡" in val:
        return "background-color: #fff3cd; color: #856404; font-weight: bold"
    if "🔴" in val:
        return "background-color: #f8d7da; color: #721c24; font-weight: bold"
    return "color: #6c757d"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _try_load(source_class):
    try:
        return source_class().load()
    except FileNotFoundError:
        return None


@st.cache_data
def load_data():
    bank_df         = BankTransactionSource().load()
    psp_df          = PSPTransactionSource().load()
    merchants_df    = pd.read_parquet("data/merchants.parquet")
    applications_df = pd.read_parquet("data/loan_applications.parquet")
    webshop_df      = _try_load(WebshopOrderSource)
    marketplace_df  = _try_load(MarketplaceSource)
    accounting_df   = _try_load(AccountingSource)
    kyc_df          = _try_load(KYCSource)
    return (bank_df, psp_df, merchants_df, applications_df,
            webshop_df, marketplace_df, accounting_df, kyc_df)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.header("1. Select Merchant")
try:
    (bank_df, psp_df, merchants_df, applications_df,
     webshop_df, marketplace_df, accounting_df, kyc_df) = load_data()

    missing_sources = [
        name for name, df in {
            "Webshop": webshop_df, "Marketplace": marketplace_df,
            "Accounting": accounting_df, "KYC": kyc_df,
        }.items() if df is None
    ]
    if missing_sources:
        st.caption(
            f"ℹ️ Sources not loaded (pessimistic defaults applied): "
            f"{', '.join(missing_sources)}. Run `python -m data.synthetic_gen` to generate."
        )

    merchant_ids = merchants_df["merchant_id"].tolist()
    selected = st.selectbox("Choose a merchant:", merchant_ids)

    m_info   = merchants_df[merchants_df["merchant_id"] == selected].iloc[0]
    app_info = applications_df[applications_df["merchant_id"] == selected].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Company",    m_info["company_name"])
    col2.metric("Sector",     m_info["sector"].capitalize())
    col3.metric("Annual GMV", f"EUR {m_info['annual_gmv']:,.0f}")
    col4.metric("Requested",  f"EUR {app_info['requested_amount']:,}")

    # -----------------------------------------------------------------------
    # Step 2: Feature scorecard
    # -----------------------------------------------------------------------
    st.header("2. Underwriting Scorecard")
    if st.button("Compute Features & Run Decision"):

        feat = compute_features(
            selected, bank_df, psp_df, m_info, app_info,
            webshop_df=webshop_df,
            marketplace_df=marketplace_df,
            accounting_df=accounting_df,
            kyc_df=kyc_df,
        )
        feat_dict = feat.to_dict()

        scorecard = build_scorecard(feat_dict)

        # Summary counts
        n_green  = scorecard["Status"].str.contains("🟢").sum()
        n_amber  = scorecard["Status"].str.contains("🟡").sum()
        n_red    = scorecard["Status"].str.contains("🔴").sum()
        n_nodata = scorecard["Status"].str.contains("⚪").sum()

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("🟢 Pass",    n_green)
        s2.metric("🟡 Caution", n_amber)
        s3.metric("🔴 Flag",    n_red)
        s4.metric("⚪ No data", n_nodata)

        st.dataframe(
            scorecard.style.map(_style_status, subset=["Status"]),
            use_container_width=True,
            hide_index=True,
        )

        # -----------------------------------------------------------------------
        # Step 3: Credit decision
        # -----------------------------------------------------------------------
        st.header("3. Credit Decision")
        pol = credit_policy(feat_dict)

        decision_color = {"APPROVE": "🟢", "MANUAL_REVIEW": "🟡", "DECLINE": "🔴"}
        st.subheader(
            f"{decision_color.get(pol['decision'], '⚪')} {pol['decision']}"
        )

        if pol["decision"] != "DECLINE":
            c1, c2, c3 = st.columns(3)
            c1.metric("Max Amount",   f"EUR {pol['max_amount']:,}")
            c2.metric("Pricing Band",
                      f"{pol['pricing_band']} "
                      f"({PRICING_LABELS.get(pol['pricing_band'], '—')})")
            c3.metric("Max Tenor",    f"{pol['tenor_max_months']} months")

        st.subheader("Reason Codes & Explanations")
        if pol["explanations"]:
            for exp in pol["explanations"]:
                st.warning(exp)
        else:
            st.success("No risk flags triggered.")

        if pol["manual_review_flags"]:
            st.info(f"Manual review: {', '.join(pol['manual_review_flags'])}")

        # -----------------------------------------------------------------------
        # Step 4: Shadow ML score
        # -----------------------------------------------------------------------
        st.header("4. Shadow ML Score (Informational)")
        try:
            ml_score = score_merchant(feat_dict)
            if ml_score is not None:
                st.metric("P(Default)", f"{ml_score:.3f}")
                st.caption(
                    "Informational only — does NOT influence the decision."
                )
            else:
                st.info("Shadow model not trained yet. Run `python run_pipeline.py` first.")
        except Exception:
            st.info("Shadow model not available. Run `python run_pipeline.py` first.")

except FileNotFoundError:
    st.error(
        "Base data files not found. "
        "Run `python -m data.synthetic_gen` then `python run_pipeline.py` first."
    )
