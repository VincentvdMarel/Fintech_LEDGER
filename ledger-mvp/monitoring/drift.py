"""
monitoring/drift.py — Population Stability Index (PSI) for feature drift.
"""

import numpy as np


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
) -> float:
    """
    Compute PSI between a reference (expected) and current (actual) distribution.
    PSI < 0.1  -> stable
    PSI 0.1-0.2 -> moderate shift
    PSI > 0.2  -> significant drift -> alert
    """
    breakpoints = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        bins + 1,
    )
    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # Avoid division by zero
    expected_pct = np.clip(expected_pct, 1e-4, None)
    actual_pct = np.clip(actual_pct, 1e-4, None)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return round(float(psi), 4)


def run_drift_report(
    reference_features: dict[str, np.ndarray],
    current_features: dict[str, np.ndarray],
) -> dict:
    """Run PSI for all features and flag drift."""
    report = {}
    for feat_name in reference_features:
        if feat_name in current_features:
            psi = compute_psi(reference_features[feat_name],
                              current_features[feat_name])
            report[feat_name] = {
                "psi": psi,
                "status": (
                    "stable" if psi < 0.1
                    else "moderate_shift" if psi < 0.2
                    else "significant_drift"
                ),
            }
    return report
