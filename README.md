# Fintech — Ledger: Working-Capital Underwriting for Dutch E-Commerce SMEs

A local MVP for credit decisioning on Dutch e-commerce working-capital loans (EUR 10k–150k, 3–18 months). Built as a transparent, auditable demo of a **two-engine underwriting system** — not production code.

> **Core idea:** build a *merchant credit passport* from consent-based, multi-source data (bank/PSD2, PSPs, webshops, marketplaces, accounting, KYC) and run it through two engines — a transparent rules engine that makes the decision today, and a shadow ML model that scores in parallel but is logged only, never used to decide.

---

## Repository Structure

```
Fintech_LEDGER/
├── ledger-mvp/          # Main project (Python underwriting engine + Streamlit UI)
│   ├── app/             # Streamlit demo UI
│   ├── config.py        # Single source of truth for all thresholds and constants
│   ├── data/            # Synthetic data generation and export utilities
│   ├── decisioning/     # Decision envelope assembly + DuckDB audit logging
│   ├── docs/            # Variable design spec, architecture notes
│   ├── features/        # Feature engineering pipeline (54 features, 10 families)
│   ├── ingestion/       # Source loaders, validation, reconciliation, consent log
│   ├── models/          # Shadow ML engine (GBM, logged only — never decides)
│   ├── monitoring/      # Drift detection, fairness metrics, performance tracking
│   ├── policy/          # Rules engine: 7 knockouts + 15 scored gates
│   ├── run_pipeline.py  # End-to-end pipeline entry point
│   ├── two_engines_demo.ipynb  # Self-contained walkthrough of both engines
│   └── README.md        # Detailed project documentation
├── CLAUDE.md            # Instructions for Claude Code AI agent
├── AGENTS.md            # Instructions for all AI coding agents
└── .claude/             # Claude Code project settings
```

---

## Two-Engine Architecture

| | Engine A — Rules (live) | Engine B — Shadow ML |
|---|---|---|
| Code | `policy/credit_policy.py` | `models/train_shadow.py`, `models/score.py` |
| What | 7 hard knockouts + 15 scored gates → APPROVE / MANUAL_REVIEW / DECLINE + limit + price | GradientBoostingClassifier → calibrated P(default) |
| Role | **Decides.** Every Year-1 approval also reviewed by a credit officer. | **Logged only.** Stored against outcomes; never influences a decision. |

We make **no claim that the ML is superior** until real repayment data exists.

---

## Quick Start

```bash
cd ledger-mvp
pip install -r requirements.txt       # 1. install dependencies
python -m data.synthetic_gen          # 2. generate synthetic merchants + DuckDB
python run_pipeline.py                # 3. run feature pipeline → policy → shadow model
streamlit run app/streamlit_app.py    # 4. launch underwriting UI
```

Everything runs locally — no external services required. The DuckDB file holds the consent and decision audit log.

For a self-contained walkthrough of both engines, open **`ledger-mvp/two_engines_demo.ipynb`** and run all cells.

---

## Key Principles

- **Rules decide; ML watches.** Human review on every Year-1 approval.
- **GDPR data minimisation** — aggregated features only; explicit consent logging.
- **Cross-reconciliation** (bank ↔ PSP ↔ accounting) for fraud detection.
- **No fabricated metrics** — no ML-superiority claim before portfolio data exists.
- **Single source of truth** — all thresholds and constants in `ledger-mvp/config.py`.

---

## Built With

- Python 3.11+, pandas, scikit-learn, DuckDB, Streamlit
- Synthetic data: 1,000 Dutch e-commerce merchants across 6 data sources
- AI-assisted development: Claude Code (engine + docs), ChatGPT/Codex (scaffolding)
- Humans own all credit, financial, and investor-facing decisions

---

## Collaboration History

All meaningful changes are tracked via git commits. See `git log` for the full history.  
Agent-assisted sessions are noted in commit messages where applicable.

---

## License

Academic / demo use only. Not licensed for commercial credit decisioning.
