"""
data/export_demo_json.py — Export pre-computed demo data for the HTML investor deck.

Selects 4 representative merchants (one per decision type), computes their
15-feature scorecard and credit-policy output, then patches the
DEMO_DATA_START … DEMO_DATA_END block in ledger-deck_6.html.

Run from the ledger-mvp directory:
    python data/export_demo_json.py
"""

import json
import math
import pickle
import re
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from ingestion.bank_source import BankTransactionSource
from ingestion.psp_source import PSPTransactionSource
from ingestion.webshop_source import WebshopOrderSource
from ingestion.marketplace_source import MarketplaceSource
from ingestion.accounting_source import AccountingSource
from ingestion.kyc_source import KYCSource
from features.pipeline import compute_features
from policy.credit_policy import credit_policy
from models.train_shadow import score_merchant, ML_FEATURES


KEY_FEATURES = [
    {"name": "net_cashflow_coverage",          "label": "Cashflow Coverage Ratio",     "source": "Bank",         "type": "inverted", "green": 1.80,  "red": 1.20,  "fmt": "ratio"},
    {"name": "revenue_volatility_90d",         "label": "Revenue Volatility (90d)",    "source": "Bank",         "type": "standard", "green": 0.25,  "red": 0.45,  "fmt": "pct"},
    {"name": "overdraft_dependency",           "label": "Overdraft Dependency",        "source": "Bank",         "type": "standard", "green": 0.10,  "red": 0.50,  "fmt": "pct"},
    {"name": "days_cash_on_hand",              "label": "Days Cash on Hand",           "source": "Bank",         "type": "inverted", "green": 45,    "red": 15,    "fmt": "days"},
    {"name": "refund_rate_ltm",                "label": "Refund Rate (LTM)",           "source": "PSP",          "type": "standard", "green": 0.03,  "red": 0.08,  "fmt": "pct"},
    {"name": "chargeback_rate_ltm",            "label": "Chargeback Rate (LTM)",       "source": "PSP",          "type": "standard", "green": 0.005, "red": 0.015, "fmt": "pct"},
    {"name": "settlement_delay_p95",           "label": "Settlement Delay P95",        "source": "PSP",          "type": "standard", "green": 5,     "red": 10,    "fmt": "days"},
    {"name": "platform_concentration",         "label": "Platform Concentration",      "source": "PSP",          "type": "standard", "green": 0.40,  "red": 0.65,  "fmt": "hhi"},
    {"name": "supplier_pay_punctuality",       "label": "Supplier Pay Regularity",     "source": "Bank",         "type": "inverted", "green": 0.90,  "red": 0.70,  "fmt": "pct"},
    {"name": "ad_spend_ratio_3m",              "label": "Ad Spend Ratio (3m)",         "source": "Bank",         "type": "standard", "green": 0.15,  "red": 0.30,  "fmt": "pct"},
    {"name": "vat_punctuality",                "label": "VAT Punctuality",             "source": "Accounting",   "type": "inverted", "green": 0.90,  "red": 0.70,  "fmt": "pct"},
    {"name": "gross_margin_avg_6m",            "label": "Gross Margin (6m avg)",       "source": "Accounting",   "type": "inverted", "green": 0.35,  "red": 0.15,  "fmt": "pct"},
    {"name": "bank_psp_recon_delta",           "label": "Bank–PSP Recon Gap",     "source": "Cross-source", "type": "standard", "green": 0.05,  "red": 0.20,  "fmt": "pct"},
    {"name": "marketplace_account_health_avg", "label": "Marketplace Health Score",    "source": "Marketplace",  "type": "inverted", "green": 8.0,   "red": 5.0,   "fmt": "score"},
    {"name": "cancellation_rate_90d",          "label": "Cancellation Rate (90d)",     "source": "Webshop",      "type": "standard", "green": 0.05,  "red": 0.12,  "fmt": "pct"},
    {"name": "return_rate_90d",                "label": "Return Rate (90d)",           "source": "Webshop",      "type": "standard", "green": 0.08,  "red": 0.15,  "fmt": "pct"},
    {"name": "fulfillment_timeliness_pct",     "label": "Fulfillment On-Time Rate",    "source": "Webshop",      "type": "inverted", "green": 0.90,  "red": 0.70,  "fmt": "pct"},
    {"name": "ubo_director_match",             "label": "UBO / Director Match",        "source": "KYC",          "type": "bool",     "green": None,  "red": None,  "fmt": "bool"},
]


class _Enc(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return None if (math.isnan(float(obj)) or math.isinf(float(obj))) else float(obj)
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


def _sf(v):
    """Safe float — returns None for NaN/Inf/None."""
    if v is None: return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _fmt(raw, fmt_type):
    if fmt_type == "bool":
        if raw is None: return "—"
        return "✓ Match" if raw else "✗ Mismatch"
    v = _sf(raw)
    if v is None: return "—"
    if fmt_type == "pct":   return f"{v * 100:.1f}%"
    if fmt_type == "ratio": return f"{v:.2f}×"
    if fmt_type == "days":  return f"{v:.1f} days"
    if fmt_type == "hhi":   return f"{v:.3f}"
    if fmt_type == "score": return f"{v:.1f} / 10"
    return str(round(v, 3))


def _status(cfg, raw):
    if cfg["type"] == "bool":
        if raw is None: return "none"
        return "green" if raw else "red"
    v = _sf(raw)
    if v is None: return "none"
    g, r = cfg["green"], cfg["red"]
    if cfg["type"] == "standard":
        return "green" if v <= g else ("amber" if v <= r else "red")
    return "green" if v >= g else ("amber" if v >= r else "red")


def _thresh(cfg):
    if cfg["type"] == "bool": return "Must match"
    g, r, fmt = cfg["green"], cfg["red"], cfg["fmt"]
    def fv(v):
        if fmt == "pct":   return f"{v*100:.0f}%"
        if fmt == "ratio": return f"{v:.1f}×"
        if fmt == "days":  return f"{v:.0f}d"
        if fmt == "score": return f"{v:.0f}/10"
        return str(v)
    if cfg["type"] == "standard": return f"Green ≤ {fv(g)}  ·  Red > {fv(r)}"
    return f"Green ≥ {fv(g)}  ·  Red < {fv(r)}"


def _scorecard(feat_dict):
    return [
        {
            "name":       cfg["name"],
            "label":      cfg["label"],
            "source":     cfg["source"],
            "value":      _fmt(feat_dict.get(cfg["name"]), cfg["fmt"]),
            "status":     _status(cfg, feat_dict.get(cfg["name"])),
            "thresholds": _thresh(cfg),
        }
        for cfg in KEY_FEATURES
    ]


def _try_load(cls):
    try:    return cls().load_and_validate()
    except FileNotFoundError: return None


def _select(all_results):
    """Pick one merchant per quadrant: approve, manual, credit-decline, kyc-decline."""
    buckets = {"approve": [], "manual": [], "credit": [], "kyc": []}
    for r in all_results:
        if r["decision"] == "APPROVE":
            buckets["approve"].append(r)
        elif r["decision"] == "MANUAL_REVIEW":
            buckets["manual"].append(r)
        elif r["decision"] == "DECLINE" and "UBO_MISMATCH" in r["reason_codes"]:
            buckets["kyc"].append(r)
        else:
            buckets["credit"].append(r)

    # Sort approvals by fewest reason codes (cleanest profile first)
    buckets["approve"].sort(key=lambda x: len(x["reason_codes"]))
    buckets["manual"].sort(key=lambda x: len(x["reason_codes"]))

    LABELS = {
        "approve": "Healthy merchant",
        "manual":  "Flagged for review",
        "credit":  "Credit decline",
        "kyc":     "KYC decline",
    }
    # Plausible display names mapped to sector for each demo bucket
    DISPLAY_NAMES = {
        "approve": {"electronics": "Volta Electronics B.V.", "fashion": "Stijlvol Mode B.V.",
                    "food": "Vers & Snel B.V.", "home": "Woonsfeer B.V.",
                    "health": "Gezond Direct B.V.", "sports": "SportDirect B.V."},
        "manual":  {"electronics": "Circuitline B.V.", "fashion": "Mode & Meer B.V.",
                    "food": "Vers & Snel B.V.", "home": "Woonstijl B.V.",
                    "health": "Vitaal Online B.V.", "sports": "Sportline B.V."},
        "credit":  {"electronics": "TechPlaza B.V.", "fashion": "Modehuis Online B.V.",
                    "food": "De Versmarkt B.V.", "home": "Interieur Direkt B.V.",
                    "health": "Gezondwinkel B.V.", "sports": "Sportwarenhuis B.V."},
        "kyc":     {"electronics": "Elektronika Trade B.V.", "fashion": "Kleding Import B.V.",
                    "food": "Verse Handel B.V.", "home": "Woonsfeer B.V.",
                    "health": "Health Import B.V.", "sports": "Sport Import B.V."},
    }
    selected = []
    for key in ("approve", "manual", "credit", "kyc"):
        if buckets[key]:
            r = {k: v for k, v in buckets[key][0].items() if k != "merchant_id"}
            r["scenario"] = LABELS[key]
            sector = r.get("sector", "").lower()
            r["company_name"] = DISPLAY_NAMES[key].get(sector, r["company_name"])
            selected.append(r)
    return selected


def main():
    root = Path(__file__).parent.parent
    os.chdir(root)

    # HTML lives one level above ledger-mvp
    html_path = root.parent / "ledger-deck_6.html"

    print("=" * 60)
    print("  Ledger — HTML Demo Data Export")
    print("=" * 60)

    print("\n[1/4] Loading data sources...")
    bank_df         = BankTransactionSource().load_and_validate()
    psp_df          = PSPTransactionSource().load_and_validate()
    merchants_df    = pd.read_parquet("data/merchants.parquet")
    applications_df = pd.read_parquet("data/loan_applications.parquet")
    webshop_df      = _try_load(WebshopOrderSource)
    marketplace_df  = _try_load(MarketplaceSource)
    accounting_df   = _try_load(AccountingSource)
    kyc_df          = _try_load(KYCSource)

    # Load shadow model once (if available) and extract global feature importances
    _shadow_model = None
    ml_top_features = []
    model_path = Path(config.SHADOW_MODEL_PATH)
    if model_path.exists():
        with open(model_path, "rb") as fh:
            _shadow_model = pickle.load(fh)
        # Human-readable labels — KEY_FEATURES covers the 15 scorecard ones; rest use title-cased name
        _kf_labels = {cfg["name"]: cfg["label"] for cfg in KEY_FEATURES}
        importances = list(zip(ML_FEATURES, _shadow_model.feature_importances_))
        importances.sort(key=lambda x: x[1], reverse=True)
        ml_top_features = [
            {
                "name":       feat_name,
                "label":      _kf_labels.get(feat_name, feat_name.replace("_", " ").title()),
                "importance": round(float(imp) * 100, 1),
            }
            for feat_name, imp in importances[:8]
        ]
        print(f"  -> Shadow model loaded. Top driver: {ml_top_features[0]['label']}")
    else:
        print("  -> Shadow model not found — run python run_pipeline.py first")

    print(f"\n[2/4] Computing features for {len(merchants_df)} merchants...")
    all_results = []
    for _, app in applications_df.iterrows():
        mid    = app["merchant_id"]
        m_info = merchants_df[merchants_df["merchant_id"] == mid].iloc[0]
        feat   = compute_features(
            mid, bank_df, psp_df, m_info, app,
            webshop_df=webshop_df, marketplace_df=marketplace_df,
            accounting_df=accounting_df, kyc_df=kyc_df,
        )
        feat_dict = feat.to_dict()
        pol      = credit_policy(feat_dict)
        ml_score = score_merchant(feat_dict, model=_shadow_model)
        all_results.append({
            "merchant_id":            mid,
            "company_name":           str(m_info["company_name"]),
            "sector":                 str(m_info["sector"]),
            "annual_gmv":             float(m_info["annual_gmv"]),
            "requested_amount":       float(app["requested_amount"]),
            "requested_tenor_months": int(app.get("requested_tenor_months", 6)),
            "decision":               pol["decision"],
            "max_amount":             float(pol["max_amount"]),
            "pricing_band":           pol["pricing_band"],
            "tenor_max_months":       int(pol["tenor_max_months"]),
            "reason_codes":           pol["reason_codes"],
            "explanations":           pol["explanations"],
            "manual_review_flags":    pol["manual_review_flags"],
            "scorecard":              _scorecard(feat_dict),
            "ml_score":               _sf(ml_score),
            "ml_top_features":        ml_top_features,
        })

    print("\n[3/4] Selecting representative merchants...")
    selected = _select(all_results)
    for m in selected:
        print(f"  [{m['decision']:<15}] {m['scenario']}: {m['company_name']}")

    # Serialise
    data_js = (
        "// DEMO_DATA_START\n"
        "  const DEMO_MERCHANTS = "
        + json.dumps(selected, indent=2, cls=_Enc, ensure_ascii=False).replace("\n", "\n  ")
        + ";\n  // DEMO_DATA_END"
    )

    print(f"\n[4/4] Patching HTML...")
    if not html_path.exists():
        print(f"  HTML not found at {html_path}")
        print("  Writing app/demo_merchants_debug.json instead.")
        with open("app/demo_merchants_debug.json", "w", encoding="utf-8") as fh:
            json.dump(selected, fh, indent=2, cls=_Enc, ensure_ascii=False)
        return

    html = html_path.read_text(encoding="utf-8")
    pattern = r"// DEMO_DATA_START\s*\n.*?// DEMO_DATA_END"
    if "// DEMO_DATA_START" not in html:
        print("  ERROR: DEMO_DATA_START marker missing. Apply the HTML template first.")
        return

    new_html = re.sub(pattern, data_js, html, flags=re.DOTALL)
    html_path.write_text(new_html, encoding="utf-8")
    print(f"  Patched {len(selected)} merchants into {html_path.name}")
    print("\n" + "=" * 60)
    print("  Done. Open ledger-deck_6.html in a browser.")
    print("=" * 60)


if __name__ == "__main__":
    main()
