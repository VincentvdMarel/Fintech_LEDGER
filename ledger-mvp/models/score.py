"""
models/score.py — Scoring utility for the shadow model.
Provides batch scoring and score explanation helpers.
"""

import pandas as pd
import numpy as np
from models.train_shadow import ML_FEATURES, score_merchant


def batch_score(feature_df: pd.DataFrame, model=None) -> pd.Series:
    """
    Score all merchants in a feature DataFrame.
    Returns a Series of shadow ML scores indexed by merchant_id.
    """
    scores = []
    for _, row in feature_df.iterrows():
        feat_dict = row.to_dict()
        score = score_merchant(feat_dict, model=model)
        scores.append({
            "merchant_id": row["merchant_id"],
            "shadow_ml_score": score,
        })
    return pd.DataFrame(scores).set_index("merchant_id")["shadow_ml_score"]


def explain_score(features: dict, model) -> dict:
    """
    Provide a basic feature-contribution explanation for a shadow score.
    Uses feature importances as a proxy (not SHAP — kept simple for MVP).
    """
    if not hasattr(model, "feature_importances_"):
        return {"note": "Feature importances not available for this model type."}

    importances = dict(zip(ML_FEATURES, model.feature_importances_))

    # Rank features by importance, annotate with actual values
    ranked = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    explanation = []
    for feat_name, importance in ranked[:5]:  # top 5
        explanation.append({
            "feature": feat_name,
            "importance": round(importance, 4),
            "value": features.get(feat_name, None),
        })

    return {
        "top_drivers": explanation,
        "note": "Feature importances are global, not per-prediction. "
                "SHAP or LIME should be added for production explanations.",
    }
