# Ledger MVP — Dutch E-Commerce SME Underwriting System

## Overview
Local MVP demonstrating end-to-end credit decisioning for Dutch e-commerce
working-capital loans (EUR 10k-150k, 3-18 months). Built for demo + foundation,
not production.

## Architecture
```
Merchant Data (PSD2 + PSP) -> Ingestion & Validation -> Feature Engineering
    -> Champion Policy (rules-based) -> Decision Envelope + Explanations
    -> Shadow ML (GBM, scoring only) -> Monitoring & Governance
```

## Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic data
python -m data.synthetic_gen

# 3. Run the full pipeline
python run_pipeline.py

# 4. Launch demo UI (optional)
streamlit run app/streamlit_app.py
```

## Key Principles
- **Phase 1:** Rules-based champion model; every loan reviewed by credit officer
- **Phase 2:** Shadow ML produces scores only — no automated decisions
- **No ML superiority claims** without real repayment data
- **GDPR data minimization** and explicit consent logging
- **Cross-reconciliation** between bank and PSP data for fraud detection

## Repository Structure
| Folder | Purpose |
|---|---|
| `data/` | Synthetic data generation and storage |
| `ingestion/` | Data loading, validation, reconciliation, consent |
| `features/` | Feature engineering pipeline (15 core features) |
| `policy/` | Rules-based champion credit policy |
| `models/` | Shadow ML training, scoring, model card |
| `decisioning/` | Final decision assembly + audit logging |
| `monitoring/` | Drift, fairness, metrics |
| `app/` | Streamlit demo UI |
| `notebooks/` | Exploration notebooks |
