# Risk Parity / Equal Risk Contribution (ERC) Portfolio Optimization Report

**Generated:** 2025-12-10 09:41:29

## Methodology Overview

Risk Parity (Equal Risk Contribution) portfolio optimization aims to equalize the risk contribution of each asset in the portfolio.
Unlike traditional mean-variance optimization, ERC focuses on risk diversification rather than return maximization.

## Risk Parity Theory

### Key Concepts

1. **Risk Contribution:** The contribution of each asset to portfolio risk
   - RC_i = w_i * (Σ * w)_i / σ_p
   - Where w_i is weight, Σ is covariance matrix, σ_p is portfolio volatility

2. **Equal Risk Contribution:** Each asset contributes equally to portfolio risk
   - RC_i = RC_j for all assets i, j
   - Target: RC_i = σ_p / n (where n is number of assets)

3. **Benefits:**
   - Better risk diversification
   - Reduced concentration risk
   - More stable portfolio structure

## Equal Risk Contribution Framework

### Optimization Objective

Minimize: sum((RC_i - target_RC)^2)

Where target_RC = σ_p / n (equal risk contribution per asset)

- **Generalized RP Enabled:** True
- **Target Risk Contributions:** equal

## Covariance Estimation Methods

- **Method:** sample
- **Estimation Windows:** [252, 500, 750]
- **Use Shrinkage:** True
- **Shrinkage Method:** ledoit_wolf

## Optimization Procedure

- **Solver:** nonlinear_programming
- **Backend:** scipy_slsqp
- **Tolerance:** 1e-08
- **Max Iterations:** 5000

### Constraints
- **Long Only:** True
- **Fully Invested:** True
- **Weight Bounds:** [0.0, 1.0]
- **Max Weight Per Asset:** 0.25

## Risk Contribution Results

- **Average Risk Parity Deviation Score:** 0.202935
  - Lower values indicate better equal risk contribution
- **Average Risk Contribution CV:** 0.202935
  - Lower values indicate more equal risk contributions
- **Average Equal Risk Gap:** 0.000131
- **Average Max Risk Contribution:** 0.000978
- **Average Min Risk Contribution:** 0.000445
- **Average Range:** 0.000533

## Portfolio Quality Metrics

- **Average Expected Return (Annualized):** 41.9378
- **Average Volatility (Annualized):** 0.1196
- **Average Sharpe Ratio:** 1.0209
- **Maximum Sharpe Ratio:** 1.0209
- **Average Sortino Ratio:** 0.9480
- **Average Max Drawdown:** 0.3764
- **Average Calmar Ratio:** 0.4421

## Comparison with Baseline Portfolios

- **Average Volatility Reduction vs Baseline:** 0.1345 (13.45%)
- **Average Sharpe Improvement vs Baseline:** 0.1527
- **ERC vs Equal Weight Volatility Difference:** 0.043367
- **ERC vs Equal Weight Sharpe Difference:** 0.0000

## Robustness and Sensitivity Analysis

### Covariance Window Sensitivity

Different estimation windows were tested to assess robustness of ERC portfolios.

- **Average Covariance Estimation Time:** 1.36 ms

## Portfolio Structure Effects

- **Average Number of Assets:** 10.0
- **Average HHI Concentration:** 0.1000
  - Lower values indicate less concentration
- **Average Effective Number of Assets:** 10.00
- **Average Weight Entropy:** 2.3026
  - Higher values indicate more diversification

## Computational Performance

- **Average Runtime per Optimization:** 20.28 ms
- **95th Percentile Runtime:** 31.78 ms
- **Average Solver Time:** 2.15 ms
- **Average Risk Contribution Calculation Time:** 0.04 ms

## Key Insights

⚠ ERC portfolios show room for improvement in risk parity
✓ ERC portfolios achieve strong risk-adjusted returns (Sharpe: 1.02)
✓ ERC portfolios maintain low volatility (11.96%)
✓ ERC portfolios show good diversification (10.0 effective assets)
