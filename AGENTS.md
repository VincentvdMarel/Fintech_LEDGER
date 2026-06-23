# AGENTS.md — AI Agent Instructions

This file provides context and operating rules for AI coding agents (Claude, Codex, Gemini, Copilot, and others) working in this repository.

## What This Project Is

**Ledger MVP** is a credit-decisioning engine for Dutch e-commerce working-capital loans.  
Location: `ledger-mvp/` subdirectory.

It implements a **two-engine underwriting architecture**:
- **Engine A** (live, rules-based): 7 hard knockouts + 15 scored gates → APPROVE / MANUAL_REVIEW / DECLINE
- **Engine B** (shadow, ML-based): GBM classifier that scores in parallel, is logged only, and **never influences a decision**

## Directory Map

| Path | Purpose |
|---|---|
| `ledger-mvp/config.py` | **Single source of truth** for all thresholds and constants |
| `ledger-mvp/ingestion/` | Data loaders for 6 sources (bank, PSP, webshop, marketplace, accounting, KYC) |
| `ledger-mvp/features/pipeline.py` | 54-feature engineering pipeline |
| `ledger-mvp/policy/` | Engine A — credit policy, knockouts, reason codes |
| `ledger-mvp/models/` | Engine B — shadow GBM training and scoring |
| `ledger-mvp/decisioning/` | Decision envelope assembly + DuckDB audit log |
| `ledger-mvp/monitoring/` | Drift detection, fairness metrics, performance tracking |
| `ledger-mvp/app/streamlit_app.py` | Streamlit underwriting UI |
| `ledger-mvp/data/synthetic_gen.py` | Generates 1,000 synthetic Dutch e-commerce merchants |
| `ledger-mvp/two_engines_demo.ipynb` | Self-contained walkthrough — run this to see both engines in action |

## Critical Rules for All Agents

1. **Never hard-code thresholds** — all constants belong in `ledger-mvp/config.py`
2. **Engine B never decides** — shadow ML scores are logged only; do not wire them into the credit decision
3. **No fabricated metrics** — do not claim ML accuracy or superiority without real repayment data
4. **Confirm before changing `config.py`** — every value there is a credit policy parameter with financial implications
5. **Do not commit data files** — `data/*.parquet` and `data/*.duckdb` are gitignored; regenerate locally
6. **Do not commit secrets or PII** — no `.env` files, API keys, or real merchant data

## How to Run

```bash
cd ledger-mvp
pip install -r requirements.txt
python -m data.synthetic_gen          # step 1: generate synthetic data
python run_pipeline.py                # step 2: run full pipeline
streamlit run app/streamlit_app.py    # step 3: launch UI
```

For a self-contained demo without Streamlit, open `two_engines_demo.ipynb` and run all cells.

## Code Conventions

- Python 3.11+
- Comments explain *why*, not *what* — the code is the what
- One-line module docstring at the top of each file is sufficient
- Follow the existing style in the file you are editing

## Architecture Invariants

- `config.py` is the single source of truth — no threshold may live elsewhere
- All data ingestion goes through consent logging (`ingestion/consent_log.py`)
- Every Year-1 approval triggers mandatory human credit-officer review
- Cross-source reconciliation (bank ↔ PSP ↔ accounting) is run for every application

## Agent Attribution

AI-assisted sessions should be noted in commit messages. Humans own all credit, financial, and investor-facing decisions.
