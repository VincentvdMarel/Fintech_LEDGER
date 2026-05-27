"""
models/train_shadow.py — Train and score with the shadow ML model.

WARNING: The shadow model produces scores ONLY. It does not make lending decisions.
"The plan should not claim that machine learning is superior before
 portfolio data exist." — Business Plan, Section 3.2
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Features used by the shadow ML model.
# Subset of all computed features — chosen for predictive value and stability.
# Keep in sync with features/pipeline.py and features/missing_handler.py.
ML_FEATURES = [
    # Cashflow
    "monthly_net_cashflow_avg_3m",
    "monthly_net_cashflow_avg_6m",
    "cashflow_trend_delta",
    "revenue_volatility_30d",
    "revenue_volatility_90d",
    "revenue_volatility_180d",
    "revenue_volatility_delta",
    "net_cashflow_coverage",
    "negative_balance_pct_90d",
    "overdraft_dependency",
    "days_cash_on_hand",
    "inflow_concentration_hhi",
    # Settlement & payments
    "settlement_delay_p95",
    "settlement_delay_median",
    "settlement_timing_variability",
    "supplier_pay_punctuality",
    "supplier_payout_lumpiness",
    # Refunds & chargebacks
    "refund_rate_30d",
    "refund_rate_ltm",
    "chargeback_rate_30d",
    "chargeback_rate_ltm",
    "refund_trend_3m",
    # Concentration & GPV
    "platform_concentration",
    "seasonality_index",
    "gpv_trend_90d",
    "payment_method_card_pct",
    # Operational
    "ad_spend_ratio_3m",
    # Reconciliation & quality
    "bank_psp_recon_delta",
    "data_coverage_score",
    # Merchant info
    "monthly_gmv_avg_6m",
    "trading_months",
]


def prepare_training_data(
    feature_df: pd.DataFrame,
    merchants_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare X (features) and y (target) for shadow model training.

    Target: 'is_bad_synthetic' from merchant table — a synthetic label
    that proxies 30+ DPD default. In production, this would be replaced
    with actual repayment outcomes.
    """
    merged = feature_df.merge(
        merchants_df[["merchant_id", "is_bad_synthetic"]],
        on="merchant_id",
        how="inner",
    )
    X = merged[ML_FEATURES].fillna(0)  # simple imputation for MVP
    y = merged["is_bad_synthetic"].astype(int)
    return X, y


def train_shadow_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "gbm",
) -> dict:
    """
    Train a shadow model and return model + evaluation metrics.

    Parameters
    ----------
    X : pd.DataFrame — feature matrix
    y : pd.Series — binary target (1 = default)
    model_type : str — "gbm" or "logistic"

    Returns
    -------
    dict with keys: model, auc_cv, brier_cv, feature_importances
    """
    if model_type == "gbm":
        model = GradientBoostingClassifier(
            n_estimators=config.SHADOW_N_ESTIMATORS,
            max_depth=config.SHADOW_MAX_DEPTH,
            learning_rate=config.SHADOW_LEARNING_RATE,
            random_state=42,
        )
    elif model_type == "logistic":
        model = LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Cross-validated evaluation
    auc_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
    brier_scores = cross_val_score(
        model, X, y, cv=5, scoring="neg_brier_score"
    )

    # Fit on full data for scoring
    model.fit(X, y)

    # Feature importances
    if hasattr(model, "feature_importances_"):
        importances = dict(zip(ML_FEATURES, model.feature_importances_))
    else:
        importances = dict(zip(ML_FEATURES, abs(model.coef_[0])))

    # Persist model
    model_path = Path(config.SHADOW_MODEL_PATH)
    model_path.parent.mkdir(exist_ok=True)
    with open(model_path, "wb") as fh:
        pickle.dump(model, fh)

    return {
        "model": model,
        "auc_cv_mean": round(np.mean(auc_scores), 4),
        "auc_cv_std": round(np.std(auc_scores), 4),
        "brier_cv_mean": round(-np.mean(brier_scores), 4),
        "feature_importances": importances,
        "n_samples": len(X),
        "n_positives": int(y.sum()),
        "model_type": model_type,
        "note": "SHADOW ONLY — not used for automated decisions",
    }


def score_merchant(features: dict, model=None) -> float | None:
    """
    Score a single merchant using the shadow model.
    Returns probability of default (0-1) or None if model not available.
    """
    
    if model is None:
        model_path = Path(config.SHADOW_MODEL_PATH)
        if not model_path.exists():
            return None
        with open(model_path, "rb") as fh:
            model = pickle.load(fh)

    X = pd.DataFrame([{k: features.get(k, 0) for k in ML_FEATURES}])
    return float(model.predict_proba(X)[0, 1])
