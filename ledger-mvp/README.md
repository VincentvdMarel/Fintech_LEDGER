# Ledger - Working-Capital Underwriting for Dutch E-Commerce SMEs

Local MVP for credit decisioning on Dutch e-commerce working-capital loans
(EUR 10k-150k, 3-18 months). Built as a demo and foundation - **not** production.

The idea: build a **merchant credit passport** from consent-based, multi-source data
(bank/PSD2, PSPs, webshops, marketplaces, accounting, KYC) and run it through **two
engines** - a transparent rules engine that makes the decision today, and a shadow ML
model that scores in parallel but is logged only, never used to decide.

## Two engines

|  | Engine A - rules (live) | Engine B - shadow ML |
|---|---|---|
| Code | `policy/credit_policy.py` | `models/train_shadow.py`, `models/score.py` |
| What | 7 hard knockouts + 15 scored gates -> APPROVE / MANUAL_REVIEW / DECLINE + limit + price | GradientBoostingClassifier -> calibrated P(default) |
| Role | **Decides.** Every Year-1 approval is also reviewed by a credit officer. | **Logged only.** Stored against outcomes; never influences a decision. |

We make **no claim that the ML is superior** until real repayment data exists.

## Architecture

```
Merchant data (PSD2 + PSP + ...)  ->  ingestion/  (validate, reconcile, consent)
   ->  features/pipeline.py        (54 features, 10 families)
   ->  policy/credit_policy.py     (Engine A - decides)
   ->  models/train_shadow.py      (Engine B - shadow score, logged)
   ->  decisioning/decision_engine.py  (envelope + DuckDB audit log)
   ->  monitoring/                 (drift, fairness, metrics)
```

## Quick start

```bash
pip install -r requirements.txt      # 1. dependencies
python -m data.synthetic_gen         # 2. generate synthetic merchants + DuckDB
python run_pipeline.py               # 3. features -> policy -> shadow model
streamlit run app/streamlit_app.py   # 4. launch the underwriting UI
```

Everything runs locally; no external services. The DuckDB file holds the consent and
decision audit log.

### Executable walkthrough (both engines, one notebook)

For a single self-contained demo that runs top-to-bottom, open
**`two_engines_demo.ipynb`** and `Run All`. It implements the rules engine and trains
the shadow ML model on synthetic data, prints every decision and shadow score, and
renders feature-importance and rules-vs-shadow charts - no DuckDB or Streamlit needed
(requires `matplotlib`, included in requirements). The notebook ships with outputs
already embedded, so the code and its output are visible without running anything.

## Features

- **54 ML features across 10 source families** (`features/pipeline.py`, listed in
  `models/train_shadow.py` as `ML_FEATURES`).
- **15 of those are scored rule-gates** used by Engine A. Thresholds live in
  `config.py` (single source of truth); reason codes in `policy/reason_codes.py`.

## Repository structure

| Path | Purpose |
|---|---|
| `ingestion/` | Source loaders, validation, reconciliation, consent log |
| `features/` | Feature engineering pipeline + missing-data handling |
| `policy/` | Engine A - rules policy, knockouts, reason codes |
| `models/` | Engine B - shadow ML training, scoring, model card |
| `decisioning/` | Decision envelope assembly + DuckDB audit logging |
| `monitoring/` | Drift, fairness, metrics |
| `app/` | Streamlit demo UI |
| `data/` | Synthetic data generation; `export_demo_json.py` feeds the deck |
| `notebooks/` | Exploration scripts |
| `docs/` | Variable design spec, architecture summary, blueprint |
| `ledger-deck_10.html` | Investor / architecture pitch deck (open in a browser) |
| `two_engines_demo.ipynb` | Self-contained executable walkthrough of both engines |

## Key principles

- Rules decide; ML watches. Human review on every Year-1 approval.
- GDPR data minimization - aggregated features, explicit consent logging.
- Cross-reconciliation (bank <-> PSP <-> accounting) for fraud detection.
- No fabricated metrics; no ML-superiority claim before portfolio data.

## Agents

Built with **Claude Code** (the Python engine + docs) and **ChatGPT / Codex**
(synthetic data + scaffolding); humans own all credit, financial, and
investor-facing decisions. See `CLAUDE.md` for the operating rules.
