# CLAUDE.md — Agent Instructions for Claude Code

This file governs how Claude Code operates within this repository.

## Project Overview

**Ledger MVP** is a credit-decisioning engine for Dutch e-commerce working-capital loans (EUR 10k–150k, 3–18 months). The project lives in `ledger-mvp/` and implements a two-engine underwriting architecture:

- **Engine A** (live): Rules-based policy — 7 hard knockouts + 15 scored gates → APPROVE / MANUAL_REVIEW / DECLINE
- **Engine B** (shadow): GradientBoostingClassifier — scores in parallel, logged only, never decides

## Repository Layout

```
ledger-mvp/
├── config.py          ← SINGLE SOURCE OF TRUTH for all thresholds; edit here only
├── ingestion/         ← Data loaders (bank, PSP, webshop, marketplace, accounting, KYC)
├── features/pipeline.py ← 54 features across 10 source families
├── policy/            ← Engine A: credit_policy.py, knockouts.py, reason_codes.py
├── models/            ← Engine B: train_shadow.py, score.py, model_card.md
├── decisioning/       ← Decision envelope + DuckDB audit log
├── monitoring/        ← Drift, fairness, metrics
├── app/               ← Streamlit UI (streamlit_app.py)
├── data/              ← Synthetic data generation (synthetic_gen.py, export_demo_json.py)
├── docs/              ← variable_design_spec.md, architecture notes
└── two_engines_demo.ipynb ← Self-contained executable walkthrough
```

## Operating Rules

### What you may do freely
- Read any file in the repository
- Edit Python source files in `ingestion/`, `features/`, `policy/`, `models/`, `decisioning/`, `monitoring/`, `app/`, `data/`
- Update `README.md`, `CLAUDE.md`, `AGENTS.md`, and files in `docs/`
- Run `python run_pipeline.py` and `python -m data.synthetic_gen` to test changes
- Run `streamlit run app/streamlit_app.py` to verify UI changes

### What requires explicit user confirmation
- Changing any threshold or constant in `config.py` — these are credit policy parameters
- Modifying the DuckDB schema in `decisioning/decision_engine.py`
- Adding or removing features from `features/pipeline.py` ML_FEATURES list
- Changing the shadow model type or training procedure in `models/train_shadow.py`
- Force-pushing or rewriting git history

### What you must never do
- Hard-code thresholds anywhere other than `config.py`
- Allow Engine B (shadow ML) to influence the credit decision — it is logged only
- Fabricate performance metrics or claim ML superiority without real portfolio data
- Commit `.env` files, credentials, or real merchant PII

## Code Style

- Python 3.11+; follow existing style in each module
- Comments explain WHY, not WHAT — well-named identifiers handle the what
- No docstrings beyond a one-line module summary at the top of each file
- All thresholds and constants → `config.py`, never inline

## Running the Project

```bash
cd ledger-mvp
pip install -r requirements.txt
python -m data.synthetic_gen          # generates synthetic data + DuckDB
python run_pipeline.py                # full pipeline run
streamlit run app/streamlit_app.py    # UI
```

## Key Invariants

1. `config.py` is the single source of truth — no threshold lives anywhere else
2. Engine A (rules) decides; Engine B (shadow ML) only logs
3. Every Year-1 approval triggers a human credit-officer review
4. All ingestion goes through consent logging (`ingestion/consent_log.py`)
5. `data/*.parquet` and `data/*.duckdb` are gitignored — regenerate locally
