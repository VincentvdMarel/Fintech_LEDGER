"""
monitoring/fairness.py — Adverse impact analysis for SME credit.

Dutch SME credit does not use protected personal characteristics.
However, proxy discrimination through sector, platform, or loan size
must be monitored.
"""

import pandas as pd


def adverse_impact_by_sector(
    decisions_df: pd.DataFrame,
    merchants_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute approval rate by sector to detect potential proxy discrimination.

    A sector with significantly lower approval rates may warrant review,
    even if the difference is driven by legitimate credit signals.
    """
    merged = decisions_df.merge(
        merchants_df[["merchant_id", "sector"]],
        on="merchant_id",
        how="left",
    )

    sector_stats = (
        merged.groupby("sector")
        .agg(
            total=("decision", "count"),
            approved=("decision", lambda x: (x.isin(["APPROVE", "MANUAL_REVIEW"])).sum()),
            declined=("decision", lambda x: (x == "DECLINE").sum()),
            avg_ml_score=("shadow_ml_score", "mean"),
        )
        .assign(
            approval_rate=lambda df: (df["approved"] / df["total"]).round(3)
        )
    )

    # Flag sectors where approval rate is < 80% of the overall rate
    overall_rate = sector_stats["approved"].sum() / sector_stats["total"].sum()
    sector_stats["flag"] = sector_stats["approval_rate"].apply(
        lambda r: "REVIEW" if r < overall_rate * 0.8 else "OK"
    )

    return sector_stats


def adverse_impact_by_platform_count(
    decisions_df: pd.DataFrame,
    merchants_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Check if single-platform merchants are disproportionately disadvantaged.

    Platform concentration (HHI = 1.0 for single-PSP) is a legitimate
    credit signal, but should be documented and reviewed.
    """
    merged = decisions_df.merge(
        merchants_df[["merchant_id", "n_psps"]],
        on="merchant_id",
        how="left",
    )

    psp_stats = (
        merged.groupby("n_psps")
        .agg(
            total=("decision", "count"),
            approved=("decision", lambda x: (x.isin(["APPROVE", "MANUAL_REVIEW"])).sum()),
        )
        .assign(
            approval_rate=lambda df: (df["approved"] / df["total"]).round(3)
        )
    )

    return psp_stats
