"""
run_pipeline.py — End-to-end MVP pipeline.
Run: python run_pipeline.py
"""

import pandas as pd
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingestion.bank_source import BankTransactionSource
from ingestion.psp_source import PSPTransactionSource
from ingestion.reconciliation import reconcile_bank_psp
from features.pipeline import compute_features
from policy.credit_policy import credit_policy
from models.train_shadow import (
    prepare_training_data, train_shadow_model, score_merchant,
)
from decisioning.decision_engine import make_decision


def main():
    print("=" * 60)
    print("  LEDGER MVP — Credit Decision Pipeline")
    print("=" * 60)

    # 1. Load data
    print("\n[1/6] Loading data...")
    bank_df = BankTransactionSource().load_and_validate()
    psp_df = PSPTransactionSource().load_and_validate()
    merchants_df = pd.read_parquet("data/merchants.parquet")
    applications_df = pd.read_parquet("data/loan_applications.parquet")
    print(f"  -> {len(merchants_df)} merchants, {len(bank_df)} bank txns, "
          f"{len(psp_df)} PSP txns")

    # 2. Compute features for all merchants
    print("\n[2/6] Computing features...")
    feature_rows = []
    for _, app in applications_df.iterrows():
        mid = app["merchant_id"]
        m_info = merchants_df[merchants_df["merchant_id"] == mid].iloc[0]
        feat = compute_features(mid, bank_df, psp_df, m_info, app)
        feature_rows.append(feat)
    feature_df = pd.DataFrame(feature_rows)
    print(f"  -> {len(feature_df)} feature vectors computed")

    # 3. Train shadow model
    print("\n[3/6] Training shadow ML model...")
    X, y = prepare_training_data(feature_df, merchants_df)
    shadow_result = train_shadow_model(X, y, model_type="gbm")
    print(f"  -> AUC (5-fold CV): {shadow_result['auc_cv_mean']:.3f} "
          f"+/- {shadow_result['auc_cv_std']:.3f}")
    print(f"  -> WARNING: Shadow model is INFORMATIONAL ONLY")

    # 4. Run credit policy + decisions
    print("\n[4/6] Running credit policy...")
    decisions = []
    for _, row in feature_df.iterrows():
        mid = row["merchant_id"]
        feat_dict = row.to_dict()

        # Champion policy
        pol = credit_policy(feat_dict)

        # Shadow ML score
        ml_score = score_merchant(feat_dict, model=shadow_result["model"])

        # Final decision envelope
        envelope = make_decision(mid, feat_dict, pol, ml_score)
        decisions.append(envelope)

    decisions_df = pd.DataFrame(decisions)

    # 5. Summary statistics
    print("\n[5/6] Decision summary:")
    print(decisions_df["decision"].value_counts().to_string())
    approved = decisions_df[decisions_df["decision"] == "APPROVE"]
    if len(approved) > 0:
        print(f"\n  Approved stats:")
        print(f"    Avg max amount: EUR {approved['max_amount_eur'].mean():,.0f}")
        print(f"    Pricing bands:  {approved['pricing_band'].value_counts().to_dict()}")
        print(f"    Avg ML score:   {approved['shadow_ml_score'].mean():.3f}")

    # 6. Example decision
    print("\n[6/6] Example decision envelope:")
    example = decisions[0]
    for key, val in example.items():
        print(f"    {key}: {val}")

    print("\n" + "=" * 60)
    print("  Pipeline complete. Decisions logged to DuckDB.")
    print("=" * 60)


if __name__ == "__main__":
    main()
