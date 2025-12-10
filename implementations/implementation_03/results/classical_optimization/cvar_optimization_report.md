# CVaR Portfolio Optimization Report

**Generated:** 2025-12-10 02:35:48

## Methodology Overview

### Conditional Value at Risk (CVaR) Optimization

CVaR optimization focuses on minimizing tail risk by optimizing the expected loss beyond the Value at Risk (VaR) threshold.
This approach is particularly effective for managing extreme downside risk.

## CVaR Theory and Definition

### Value at Risk (VaR)

VaR_α(X) = -inf{x : P(X ≤ x) ≥ α}

VaR represents the maximum loss that will not be exceeded with probability α.

### Conditional Value at Risk (CVaR)

CVaR_α(X) = E[X | X ≤ VaR_α(X)]

CVaR (also known as Expected Shortfall) represents the expected loss given that the loss exceeds VaR.
CVaR is a coherent risk measure and provides better tail risk assessment than VaR alone.

## Rockafellar-Uryasev Linear Programming Formulation

The CVaR optimization problem is reformulated as a linear program:

```
minimize: λ_cvar * (VaR + (1/(1-α)) * Σ z_s / S) + λ_return * (-μ^T w)
subject to:
  z_s ≥ -portfolio_return_s - VaR  for all scenarios s
  z_s ≥ 0  for all scenarios s
  portfolio constraints
```

Where:
- α = confidence level
- S = number of scenarios
- z_s = auxiliary variables for each scenario

## Scenario Generation Procedure

- **Source:** historical
- **Estimation Windows:** [252, 500, 750] days
- **Block Bootstrap:** False

## CVaR Optimization Results

### CVaR Statistics

- **Mean CVaR:** 0.000000
- **Min CVaR:** -0.000000
- **Max CVaR:** -0.000000

### Expected Return Statistics

- **Mean Expected Return:** 0.000000
- **Min Expected Return:** 0.000000
- **Max Expected Return:** 0.000000

### CVaR-Adjusted Sharpe Ratios

- **Mean CVaR Sharpe:** nan
- **Max CVaR Sharpe:** nan

## CVaR-Return Frontier Analysis

The CVaR-return frontier shows the tradeoff between expected return and tail risk (CVaR).
Portfolios on the efficient frontier minimize CVaR for a given target return.

## Portfolio Structure Effects

- **Mean HHI Concentration:** 0.1000
- **Mean Effective Number of Assets:** 10.00

## Backtesting and Tail Risk Behavior

CVaR optimization focuses on managing extreme tail risk.
The optimized portfolios should demonstrate lower tail losses compared to mean-variance portfolios.

## Computational Performance

- **Mean Runtime:** 43.49 ms
- **Mean Solver Time:** 13.77 ms

## Key Insights

1. CVaR optimization provides superior tail risk management compared to variance-based approaches.
2. The Rockafellar-Uryasev formulation enables efficient linear programming solution.
3. Scenario-based approach captures non-normal return distributions effectively.
