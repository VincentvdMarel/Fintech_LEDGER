"""
models/score.py — Scoring utility for the shadow model.
Provides batch scoring and score explanation helpers.
"""

import pandas as pd
import numpy as np
from models.train_shadow import ML_FEATURES, score_merchant, feature_importances


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
    Uses GLOBAL feature importances as a proxy (not per-merchant SHAP).
    """
    importances = feature_importances(model)
    if not importances:
        return {"note": "Feature importances not available for this model type."}

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
        "note": "Global importance, not per-merchant SHAP. "
                "SHAP or LIME should be added for production explanations. "
                "Shadow only — does not influence the credit decision.",
    }
