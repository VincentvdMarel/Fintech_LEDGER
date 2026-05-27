"""
app/streamlit_app.py — Ledger MVP Demo UI.
Run:  streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.bank_source import BankTransactionSource
from ingestion.psp_source import PSPTransactionSource
from features.pipeline import compute_features
from policy.credit_policy import credit_policy
from models.train_shadow import score_merchant
from decisioning.decision_engine import make_decision
from monitoring.metrics import compute_portfolio_metrics

st.set_page_config(page_title="Ledger MVP", page_icon="📊", layout="wide")
st.title("📊 Ledger MVP — Credit Decision Demo")


@st.cache_data
def load_data():
    """Load all synthetic data."""
    bank_df = BankTransactionSource().load()
    psp_df = PSPTransactionSource().load()
    merchants_df = pd.read_parquet("data/merchants.parquet")
    applications_df = pd.read_parquet("data/loan_applications.parquet")
    return bank_df, psp_df, merchants_df, applications_df


# --- Step 1: Merchant selection ---
st.header("1. Select Merchant")
try:
    bank_df, psp_df, merchants_df, applications_df = load_data()
    merchant_ids = merchants_df["merchant_id"].tolist()
    selected = st.selectbox("Choose a merchant:", merchant_ids)

    m_info = merchants_df[merchants_df["merchant_id"] == selected].iloc[0]
    app_info = applications_df[applications_df["merchant_id"] == selected].iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Company", m_info["company_name"])
    col2.metric("Annual GMV", f"EUR {m_info['annual_gmv']:,.0f}")
    col3.metric("Requested", f"EUR {app_info['requested_amount']:,}")

    # --- Step 2: Feature computation ---
    st.header("2. Feature Computation")
    if st.button("Compute Features & Run Decision"):
        feat = compute_features(selected, bank_df, psp_df, m_info, app_info)
        feat_dict = feat.to_dict()

        # Display features as traffic lights
        st.subheader("Feature Vector")
        feat_display = {k: v for k, v in feat_dict.items()
                       if k not in ["merchant_id", "kvk_active", "sanctions_hit",
                                    "requested_amount", "requested_tenor_months"]}
        st.dataframe(pd.DataFrame([feat_display]).T.rename(columns={0: "Value"}))

        # --- Step 3: Policy decision ---
        st.header("3. Credit Decision")
        pol = credit_policy(feat_dict)

        # Decision card
        decision_color = {
            "APPROVE": "🟢", "MANUAL_REVIEW": "🟡", "DECLINE": "🔴"
        }
        st.subheader(f"{decision_color.get(pol['decision'], '⚪')} {pol['decision']}")

        if pol["decision"] != "DECLINE":
            c1, c2, c3 = st.columns(3)
            c1.metric("Max Amount", f"EUR {pol['max_amount']:,}")
            c2.metric("Pricing Band", f"{pol['pricing_band']} ({pol['pricing_band'] and {0.11:'11%', 0.125:'12.5%', 0.14:'14%'}.get(0.125, '12.5%')})")
            c3.metric("Max Tenor", f"{pol['tenor_max_months']} months")

        # Explanations
        st.subheader("Reason Codes & Explanations")
        for exp in pol["explanations"]:
            st.warning(exp)

        if pol["manual_review_flags"]:
            st.info(f"Manual review flags: {', '.join(pol['manual_review_flags'])}")

        # --- Step 4: Shadow ML ---
        st.header("4. Shadow ML Score (Informational)")
        try:
            ml_score = score_merchant(feat_dict)
            if ml_score is not None:
                st.metric("P(Default)", f"{ml_score:.3f}")
                st.caption("This score is informational only and does NOT influence the decision.")
            else:
                st.info("Shadow model not yet trained. Run run_pipeline.py first.")
        except Exception:
            st.info("Shadow model not available. Run run_pipeline.py first.")

except FileNotFoundError:
    st.error("Data files not found. Run `python -m data.synthetic_gen` first.")
