# Ledger Underwriting System — Variable Design Specification

## 1. Scope & Objectives

### Target Segment
Dutch e-commerce SMEs:
- Multi-platform revenue (PSPs, marketplaces, webshops)
- €0.3M–€5M GMV
- ≥2 years trading history
- Working capital use cases (inventory, supplier gaps, marketing)

### Product
- Term loans: €10k–€150k
- Tenor: 3–18 months
- Fixed amortization
- No collateral (base case)

### Primary Risk Objective
Minimize Expected Loss (EL):

EL ≈ PD × LGD × EAD

Focus:
- PD (primary early-stage target)
- LGD via dispute/refund quality
- EAD via limit & tenor structuring

---

## 2. System Constraints

- 80–120 interpretable features
- GDPR data minimization enforced
- Reconciliation-heavy underwriting file
- Phase-based rollout:

| Phase | Model Type | Characteristics |
|------|------------|----------------|
| 1 | Rules-based | Human review required |
| 2 | Shadow ML | No decisioning |
| 3 | ML | With explainability + monitoring |

---

## 3. Data Sources

A) PSD2 / Bank (AISP)  
B) PSP (Mollie, Adyen)  
C) Webshop (Shopify etc.)  
D) Marketplace (bol.com, Amazon)  
E) Accounting / Bookkeeping  
F) KYC / Fraud / Reconciliation  

---

## 4. Core Variable Families

Each variable is implemented as a **feature family**:

- Windows: 30 / 90 / 180 days + trailing 12 months
- Seasonality-adjusted variants
- Deltas (short vs long-term trend)

---

## 5. Variable Set by Data Source

---

## A) PSD2 / Bank Variables

### Key features
- Net operating cashflow trend
- Cashflow volatility (CV)
- Days cash on hand
- Overdraft dependency
- Failed payment rate
- Tax/payment irregularities
- Inflow concentration (HHI)
- Supplier payout lumpiness
- Penalty/NSF fee incidence
- Settlement lag proxy

### Risk mechanisms
- Liquidity stress → PD
- Volatility → PD
- Concentration → PD
- Payment discipline → PD

---

## B) PSP Variables

### Key features
- GPV trend
- Refund rate (value + count)
- Chargeback/dispute rate
- Dispute win-rate
- Settlement timing variability
- Authorization-to-capture ratio
- Payment method mix
- Fraud flag rate
- Buyer concentration
- Payout anomalies

### Risk mechanisms
- Refunds/chargebacks → PD + LGD
- Settlement delays → liquidity stress
- Fraud signals → PD

---

## C) Webshop Variables

### Key features
- Order volume trend
- Cancellation rate
- Return initiation rate
- Fulfillment timeliness
- Repeat customer rate
- AOV stability
- SKU concentration
- Discount intensity
- Customer support burden
- Conversion trend (if available)

### Risk mechanisms
- Demand stability → PD
- Customer satisfaction → refund/chargeback risk
- Product concentration → fragility

---

## D) Marketplace Variables

### Key features
- Marketplace GMV share
- Account health / defect score
- Payout holds/reserves
- Late shipment rate
- Return rate (channel)
- Cancellation rate
- Negative feedback rate
- Listing concentration
- Policy violations
- Payout lag variability

### Risk mechanisms
- Platform dependency → PD
- Enforcement risk → PD/LGD
- Customer satisfaction → PD/LGD

---

## E) Accounting Variables

### Key features
- Revenue ↔ GMV reconciliation ratio
- Gross margin proxy
- Current ratio (liquidity)
- Accounts payable days (AP)
- Accounts receivable days (AR)
- VAT punctuality
- Seasonality amplitude
- Expense rigidity ratio
- Owner draw leakage
- Debt service burden

### Risk mechanisms
- Margin erosion → PD/LGD
- Tax behavior → PD
- Supplier stress → PD
- Financial structure → PD

---

## F) KYC / Fraud / Reconciliation Variables

### Key features
- KvK age & status
- UBO / director consistency
- Sanctions / PEP screening
- PSP ↔ Bank reconciliation gap
- Orders ↔ Captures mismatch
- Marketplace ↔ Bank mismatch
- Identity change frequency
- Circular transaction patterns
- Data coverage score
- GDPR compliance flags

### Risk mechanisms
- Fraud → PD spike
- Data incompleteness → model risk
- Identity mismatch → decline

---

## 6. Cross-Source Reconciliation Logic

### Required reconciliations

| Flow | Must match |
|------|-----------|
| PSP payouts | Bank credits |
| Webshop orders | PSP captures |
| Marketplace payouts | Bank credits |
| Accounting revenue | GMV reconstruction |
| VAT obligations | Bank tax payments |
| Supplier invoices | Bank payments |

---

### Mismatch interpretation

| Pattern | Interpretation | Action |
|--------|--------------|--------|
| Timing lag only | Benign | Accept |
| Missing payouts | Hold/reserve/fraud | Manual review |
| Orders > captures | Fake demand | Investigate |
| Captures > orders | Hidden revenues | Reduce limit |
| Accounting mismatch | Manipulation risk | Review/decline |
| Identity mismatch | Fraud | Decline |

---

## 7. Escalation Rules

### Manual review
- Reconciliation gaps
- Sudden volatility spikes
- Data coverage gaps

### Reduce limit / tenor
- High volatility
- High concentration
- Thin liquidity

### Decline
- KYC/UBO mismatch
- Sanctions hit
- Fraud indicators
- Persistent reconciliation gaps

---

## 8. Must-Have Variables (Top Risk Drivers)

1. Bank–PSP reconciliation gap
2. Chargeback rate
3. Refund rate
4. Net operating cashflow trend
5. Cashflow volatility
6. Liquidity buffer
7. Settlement variability
8. Revenue concentration
9. VAT punctuality
10. Gross margin proxy
11. Overdraft dependency
12. Marketplace account health
13. Orders ↔ captures mismatch
14. Supplier stress
15. Identity consistency

---

## 9. Minimum Viable Underwriting File (MVUF)

### Required for approval decision

#### Identity & Compliance
- KvK
- UBO consistency
- Sanctions check

#### Reconciliation
- PSP ↔ bank
- Orders ↔ payments

#### Bank signals
- Cashflow trend
- Volatility
- Liquidity buffer
- Overdraft usage

#### Commercial signals
- Refund rate
- Chargeback rate
- Concentration

#### Bookkeeping (if available)
- VAT punctuality
- Margin proxy

#### Sizing inputs
- GMV level (90d)
- Seasonality flag

---

## 10. Phase-1 Rules-Based Policy

### Hard stops
- Sanctions / KYC failures
- Reconciliation mismatch
- Extreme dispute/refund behavior

---

### Risk tiers

| Tier | Profile |
|------|--------|
| A | Stable, diversified, low disputes |
| B | Minor weaknesses |
| C | Multiple moderate risks |
| D | Near-decline |

---

### Limit sizing (interpretive)

Base limit = min(
  % of 90d GMV,
  multiple of monthly cashflow,
  liquidity-based cap
)

Adjust down for:
- volatility
- concentration
- settlement delays

---

### Tenor rules
- High seasonality → shorter tenor
- Low liquidity → shorter tenor
- High instability → shorter tenor

---

## 11. GDPR & Data Minimization

- Use aggregated variables only
- Avoid storing raw personal data
- Prefer ratios, flags, and trends
- Hash identifiers where possible
- Enforce purpose limitation

---

## 12. Design Principles

All variables must be:

- Predictive
- Hard to manipulate
- Cross-verifiable
- Interpretable
- Stable over time
- Actionable for underwriting

---

## 13. Future Extensions (Phase 2–3)

- Gradient boosting / tree models
- Feature interaction learning
- Early warning monitoring
- Dynamic limit adjustment
- Adverse impact testing