# DeFi Wallet Credit Scoring — Aave V2

## Overview

This project implements a robust, data-driven credit scoring system for DeFi wallets interacting with the Aave V2 protocol. The model assigns a **credit score between 0 and 1000** to each wallet, based solely on historical transaction behavior.

- **Higher scores** indicate reliable, responsible usage.
- **Lower scores** reflect risky, bot-like, or exploitative behavior.

## Methodology

### 1. **Feature Engineering**
- Features are engineered from raw transaction data (`user-wallet-transactions.json`), including:
  - Total deposit/borrow in USD
  - Repayment and redemption ratios
  - Liquidation events
  - Wallet age and activity
  - Asset diversity
  - Bot/exploit signals (e.g., rapid transactions, short loans)
- Features were **initially weighted based on domain knowledge**, then **recalibrated using variance and correlation analysis** for statistical robustness.

### 2. **Normalization**
- Features are normalized using log transforms, min-max scaling, and inversion where appropriate, ensuring fair contribution to the score.

### 3. **Scoring Model**
- An **Isolation Forest** (unsupervised anomaly detection) is used to assign a risk score to each wallet.
- Scores are **inverted and scaled to 0–1000** (higher = better).

### 4. **Outputs**
- `wallet_scores.csv`: Wallet addresses and their credit scores.
- `normalized_features.csv`: Normalized feature values for transparency.
- `feature_summary.csv`: Feature definitions and normalization methods.
- `score_distribution.png`: Distribution of credit scores.

## Architecture & Processing Flow

```
Raw JSON: user-wallet-transactions.json
|
v
Feature Engineering & Normalization
|
v
Isolation Forest Scoring
|
v
Score Scaling (0-1000)
|
v
wallet_scores.csv
```

## How to Run

1. Place `user-wallet-transactions.json` in the project directory.
2. Run the analysis pipeline:
   ```bash
   python analysis.py
   ```
3. Outputs will be generated in the same directory.

## Extensibility

- The pipeline is modular and can be extended with new features or alternative models.
- All steps are transparent and reproducible for auditability.

## Files

- `analysis.py` — One-step pipeline: JSON → features → normalized → score
- `wallet_scores.csv` — Final scores
- `score_distribution.png` — Score distribution plot
- `feature_summary.csv` — Feature definitions
- `normalized_features.csv` — Normalized features
- `analysis.md` — Score analysis and behavioral insights

---
"# Defi-project" 
