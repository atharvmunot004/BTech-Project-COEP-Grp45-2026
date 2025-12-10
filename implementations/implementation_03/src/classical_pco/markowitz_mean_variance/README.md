# Markowitz Mean-Variance Portfolio Optimization

This module implements the classical Markowitz (1952) mean-variance portfolio optimization framework.

## Overview

The Markowitz mean-variance optimization solves the portfolio construction problem by balancing expected return and risk (variance). The optimization problem is:

```
minimize: w^T Σ w - λ * μ^T w
subject to: Σ w_i = 1, w_i >= 0 (long-only)
```

Where:
- `w` = portfolio weights vector
- `Σ` = covariance matrix
- `μ` = expected returns vector
- `λ` = risk aversion parameter

## Features

- **Multiple Objectives**: Minimum variance, maximum Sharpe ratio, or risk-return tradeoff
- **Covariance Estimation**: Sample covariance with optional Ledoit-Wolf shrinkage
- **Expected Returns**: Historical mean returns with annualization
- **Constraints**: Long-only, fully invested, weight bounds, max weight per asset
- **Efficient Frontier**: Generate efficient frontier portfolios
- **Comprehensive Metrics**: Portfolio quality, structure, risk, and distribution metrics
- **Performance Analysis**: Runtime metrics and optimization statistics

## Module Structure

```
markowitz_mean_variance/
├── __init__.py              # Module initialization
├── main.py                  # Main orchestration script
├── returns.py               # Data loading and returns computation
├── markowitz_optimizer.py   # Core optimization logic
├── metrics.py               # Performance metrics computation
├── report_generator.py      # Report generation
├── time_sliced_metrics.py   # Time-sliced analysis
├── llm.json                 # Configuration file
└── README.md                # This file
```

## Usage

### Basic Usage

```python
from src.classical_pco.markowitz_mean_variance import run_markowitz_optimization

# Run optimization with default config
results = run_markowitz_optimization()

# Or specify config path
results = run_markowitz_optimization(config_path='path/to/llm.json')
```

### Configuration

The module uses a JSON configuration file (`llm.json`) with the following structure:

```json
{
  "inputs": {
    "panel_price_path": "data/preprocessed/panel_price.parquet",
    "return_type": "log",
    "portfolio_universe_path": "portfolios/portfolios.parquet",
    "risk_free_rate": 0.0001
  },
  "markowitz_settings": {
    "objective": "min_variance",
    "risk_return_tradeoff": {
      "use_risk_aversion": true,
      "lambda_values": [0.1, 0.5, 1, 2, 5]
    },
    "constraints": {
      "long_only": true,
      "fully_invested": true,
      "no_short_selling": true,
      "weight_bounds": [0.0, 1.0],
      "max_weight_per_asset": 0.25,
      "min_num_assets": 1
    },
    "covariance_estimation": {
      "method": "sample",
      "estimation_windows": [252, 500, 750],
      "shrinkage": {
        "use_shrinkage": true,
        "method": "ledoit_wolf"
      }
    },
    "frontier_settings": {
      "compute_efficient_frontier": true,
      "num_portfolios": 100,
      "risk_levels": "auto"
    },
    "expected_return_estimation": {
      "method": "historical_mean",
      "use_annualization": true
    },
    "optimization_solver": {
      "method": "quadratic_programming",
      "qp_backend": "osqp",
      "tolerance": 1e-6
    },
    "random_seed": 42
  },
  "outputs": {
    "optimal_portfolios": "results/classical_optimization/markowitz_optimal_weights.parquet",
    "efficient_frontier": "results/classical_optimization/markowitz_efficient_frontier.parquet",
    "metrics_table": "results/classical_optimization/markowitz_metrics.parquet",
    "metrics_json": "results/classical_optimization/markowitz_metrics.json",
    "summary_report": "results/classical_optimization/markowitz_report.md"
  }
}
```

## Outputs

The module generates the following outputs:

1. **Optimal Portfolios** (`markowitz_optimal_weights.parquet`): Portfolio weights for optimized portfolios
2. **Efficient Frontier** (`markowitz_efficient_frontier.parquet`): Efficient frontier portfolios with risk-return characteristics
3. **Metrics Table** (`markowitz_metrics.parquet`): Comprehensive metrics DataFrame
4. **Metrics JSON** (`markowitz_metrics.json`): Structured JSON with all results
5. **Summary Report** (`markowitz_report.md`): Markdown report with analysis and insights

## Performance Metrics

### Portfolio Quality
- Expected return
- Volatility (standard deviation)
- Portfolio variance
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Calmar ratio

### Structure Metrics
- Number of assets in portfolio
- HHI concentration
- Effective number of assets
- Weight entropy
- Pairwise correlation mean

### Risk Metrics
- Value at Risk (VaR)
- Conditional Value at Risk (CVaR)
- Downside deviation

### Distribution Metrics
- Skewness
- Kurtosis
- Jarque-Bera test for normality

### Runtime Metrics
- Runtime per optimization
- 95th percentile runtime
- Covariance estimation time
- Solver time

## Dependencies

- `pandas` >= 1.5.0
- `numpy` >= 1.23.0
- `scipy` >= 1.9.0
- `cvxpy` >= 1.3.0 (for optimization)
- `osqp` >= 0.6.0 (optional, recommended solver)
- `scikit-learn` >= 1.2.0 (for Ledoit-Wolf shrinkage)
- `pyarrow` >= 10.0.0 (for Parquet support)

## References

- Markowitz, H. (1952). Portfolio Selection. *The Journal of Finance*, 7(1), 77-91.
- Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365-411.

