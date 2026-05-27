# Model Card — Ledger Shadow ML v0.1

## Model Details
- **Type:** GradientBoostingClassifier (scikit-learn)
- **Target:** Synthetic default proxy (`is_bad_synthetic`)
- **Features:** 15 interpretable features (see `ML_FEATURES` in `train_shadow.py`)
- **Training data:** 200 synthetic merchants, ~5% positive rate

## Intended Use
- Shadow scoring ONLY — no automated decisions
- Used for champion/challenger comparison and calibration
- Helps identify where rules-based policy and ML disagree

## Limitations
- Trained on synthetic data; no real repayment outcomes
- Cannot be used to claim ML outperforms champion policy
- Feature distributions may not reflect real Dutch e-commerce SMEs
- Model should be retrained once real portfolio data exists (200+ observations)

## Ethical Considerations
- No protected characteristics (gender, ethnicity, age) used as features
- Sector-level adverse impact monitored quarterly
- All decisions require human credit officer review in Year 1
- Platform concentration (HHI) is a legitimate credit signal but may
  disadvantage single-platform merchants — documented and reviewed

## Metrics (on synthetic data)
- **AUC:** [filled at training time]
- **Brier score:** [filled at training time]
- **Note:** These metrics are on synthetic data and do NOT indicate
  real-world model performance.

## Update Policy
- Retrain after 200+ real repayment observations
- Re-evaluate feature set quarterly
- Model governance review before any shift from shadow to challenger
