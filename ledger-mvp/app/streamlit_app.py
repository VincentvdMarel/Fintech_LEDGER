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

import config
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

# Pricing labels derived from config (single source of truth).
PRICING_LABELS = {k: f"{v * 100:.1f}%" for k, v in config.PRICING_BANDS.items()}

# ---------------------------------------------------------------------------
# LAYER 3 — Dashboard scorecard (DISPLAY ONLY).
# The 18 signals and their thresholds are read directly from config so the UI
# can never drift from the policy. type/fmt meaning:
#   type "standard" : higher value = worse risk
#   type "inverted" : lower  value = worse risk
#   type "bool"     : True = good, False = bad
# ---------------------------------------------------------------------------
KEY_FEATURES = config.get_dashboard_signals()


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
        if fmt == "pct":   return f"{v*100:.1f}%"   # one decimal (0.5% / 1.5%)
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
    bank_df         = BankTransactionSource().load_and_validate()
    psp_df          = PSPTransactionSource().load_and_validate()
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

        # Run policy first so signal_pass_rate is sourced from the engine
        # (15 scored-gates denominator — consistent with what the HTML deck shows).
        pol = credit_policy(feat_dict)

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

        # Signal Pass Rate — DISPLAY-ONLY health index, sourced from the policy
        # engine (green scored gates ÷ 15 total scored gates). Matches the HTML deck.
        # This number does NOT drive the decision; the policy rules do.
        st.metric("Signal Pass Rate", f"{pol['signal_pass_rate'] * 100:.0f}%")
        st.caption(
            "Signal Pass Rate is a display-only health index "
            "(green scored gates ÷ 15 total). Phase 1 does **not** approve by a "
            "numeric score — the decision below is derived purely from policy rules."
        )

        st.dataframe(
            scorecard.style.map(_style_status, subset=["Status"]),
            use_container_width=True,
            hide_index=True,
        )

        # -----------------------------------------------------------------------
        # Step 3: Credit decision
        # -----------------------------------------------------------------------
        st.header("3. Credit Decision")

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
                st.metric("Calibrated P(Default)", f"{ml_score:.3f}")
                st.caption(
                    "Shadow only — calibrated probability of default, logged for "
                    "monitoring. Does NOT influence the decision."
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
