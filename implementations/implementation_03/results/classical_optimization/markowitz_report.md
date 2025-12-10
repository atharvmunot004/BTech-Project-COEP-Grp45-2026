# Markowitz Mean-Variance Portfolio Optimization Report

**Generated:** 2025-12-10 01:59:11

## Methodology Overview

### Markowitz Mean-Variance Optimization

The Markowitz (1952) mean-variance framework optimizes portfolios by balancing expected return and risk (variance).
The optimization problem is:

```
minimize: w^T Σ w - λ * μ^T w
subject to: Σ w_i = 1, w_i >= 0 (long-only)
```

Where:
- w = portfolio weights vector
- Σ = covariance matrix
- μ = expected returns vector
- λ = risk aversion parameter

### Optimization Settings

- **Objective:** min_variance
- **Risk Aversion Parameters (λ):** [0.1, 0.5, 1, 2, 5]
- **Long Only:** True
- **Fully Invested:** True
- **Max Weight per Asset:** 0.25
- **Covariance Method:** sample
- **Estimation Windows:** [252, 500, 750] days
- **Shrinkage:** ledoit_wolf

## Mean-Variance Theory

The efficient frontier represents the set of portfolios that offer the highest expected return for a given level of risk, or the lowest risk for a given expected return.

Key concepts:
- **Efficient Frontier:** Optimal risk-return combinations
- **Minimum Variance Portfolio:** Portfolio with lowest risk
- **Maximum Sharpe Portfolio:** Portfolio with highest risk-adjusted return
- **Risk-Return Tradeoff:** Higher returns require accepting higher risk

## Summary Statistics

- **Total Portfolios Optimized:** 100

## Portfolio Optimization Results

- **Average Expected Return:** 0.148664 (14.87%)

- **Average Volatility:** 0.006915 (0.69%)

- **Average Portfolio Variance:** 0.000048

- **Average Sharpe Ratio:** 0.9373
- **Maximum Sharpe Ratio:** 0.9373

- **Average Sortino Ratio:** 0.8974

## Risk-Return Tradeoff Analysis

- **Return-Volatility Correlation:** nan
  - Positive correlation indicates higher returns come with higher risk

### Sharpe Ratio Distribution

- **Mean:** 0.9373
- **Std:** 0.0000
- **Min:** 0.9373
- **Max:** 0.9373

## Portfolio Structure Effects

- **Average Number of Assets:** 9.00

- **Average HHI Concentration:** 0.1646
  - Higher values indicate more concentrated portfolios
  - HHI ranges from 1/n (equal weights) to 1 (single asset)

- **Average Effective Number of Assets:** 6.08

- **Average Weight Entropy:** 1.9384
  - Higher entropy indicates better diversification

- **Average Pairwise Correlation:** 0.3041

## Risk Metrics

- **Average Value at Risk (VaR):** 0.015834
- **Average Conditional VaR (CVaR):** 0.025761

## Robustness and Normality Checks

- **Average Skewness:** -0.9242
  - Negative skewness indicates left tail risk

- **Average Excess Kurtosis:** 19.6177
  - Positive kurtosis indicates fat tails

- **Normality Rejection Rate (p < 0.05):** 100.00%
  - Higher rate indicates more non-normal return distributions

## Computational Performance

- **Average Runtime per Optimization:** 23.50 ms

- **Average Covariance Estimation Time:** 1.48 ms

- **Average Solver Time:** 10.19 ms

## Key Insights

- Maximum Sharpe ratio achieved: 0.9373
- Portfolios show good diversification (HHI < 0.2)
