# Variance-Covariance Value-at-Risk Evaluation

This module implements comprehensive evaluation of Variance-Covariance (Parametric) Value-at-Risk for portfolio risk assessment.

## Overview

The Variance-Covariance method assumes returns follow a normal distribution and calculates VaR as:

```
VaR = -μ - z_α * σ * √(horizon)
```

Where:
- μ = mean return
- σ = standard deviation of returns  
- z_α = z-score for confidence level α
- horizon = time horizon in days

## Features

- **Daily Returns Computation**: Computes log or simple returns from price data
- **Portfolio Returns**: Calculates portfolio returns from asset returns and weights
- **Rolling VaR**: Implements rolling window VaR calculation
- **Backtesting**: Comprehensive backtesting with violation analysis
- **Accuracy Metrics**: Hit rate, violation ratio, Kupiec test, Christoffersen test
- **Tail Risk Metrics**: Mean exceedance, max exceedance, quantile loss
- **Portfolio Structure Metrics**: HHI, effective number of assets, condition number
- **Distribution Metrics**: Skewness, kurtosis, Jarque-Bera normality test
- **Report Generation**: Comprehensive markdown reports

## Usage

### Command Line

```bash
# Use all CPU cores (default)
python -m src.classical-risk.variance-covariance-value-at-risk.main --config llm.json

# Use specific number of workers
python -m src.classical-risk.variance-covariance-value-at-risk.main --config llm.json --n-jobs 8

# Sequential processing (for debugging)
python -m src.classical-risk.variance-covariance-value-at-risk.main --config llm.json --n-jobs 1
```

### Python API

```python
from src.classical_risk.variance_covariance_value_at_risk import evaluate_var

# Load configuration and evaluate (uses all CPU cores by default)
results_df = evaluate_var(config_path='llm.json')

# Use specific number of workers
results_df = evaluate_var(config_path='llm.json', n_jobs=8)

# Sequential processing
results_df = evaluate_var(config_path='llm.json', n_jobs=1)

# Or use configuration dictionary
config = {
    'inputs': {
        'panel_price_path': 'data/processed/panel_price.parquet',
        'portfolio_weights_path': 'portfolios/portfolios.parquet'
    },
    'var_settings': {
        'confidence_levels': [0.95, 0.99],
        'horizons': [1, 10],
        'estimation_windows': [252]
    },
    # ... other settings
}

results_df = evaluate_var(config_dict=config, n_jobs=8)
```

## Configuration

The module uses a JSON configuration file (`llm.json`) with the following structure:

```json
{
    "task": "classical_var_evaluation",
    "inputs": {
        "panel_price_path": "data/processed/panel_price.parquet",
        "portfolio_weights_path": "portfolios/portfolios.parquet"
    },
    "var_settings": {
        "confidence_levels": [0.95, 0.99],
        "horizons": [1, 10],
        "estimation_windows": [252]
    },
    "modules": {
        "compute_daily_returns": true,
        "compute_portfolio_returns": true,
        "align_returns_and_var": true,
        "rolling_var": true,
        "var_violations": true,
        "accuracy_metrics": ["hit_rate", "violation_ratio", "kupiec", "christoffersen"],
        "tail_metrics": ["mean_exceedance", "max_exceedance"],
        "structure_metrics": ["hhi", "enc", "condition_number"],
        "distribution_metrics": ["skew", "kurtosis", "jarque_bera"],
        "runtime_metrics": true
    },
    "outputs": {
        "metrics_table": "results/var_metrics.parquet",
        "summary_report": "results/var_report.md"
    }
}
```

## Output

The module generates:

1. **Metrics Table** (Parquet/CSV): DataFrame with all computed metrics for each portfolio-configuration combination
2. **Summary Report** (Markdown): Comprehensive report with:
   - Methodology overview
   - Backtesting results
   - Tail risk analysis
   - Portfolio structure effects
   - Robustness and normality checks
   - Computational performance
   - Key insights

## Module Structure

- `main.py`: Main orchestration script
- `returns.py`: Returns computation (daily and portfolio)
- `var_calculator.py`: VaR calculation and rolling window implementation
- `backtesting.py`: Violation detection and accuracy metrics
- `metrics.py`: Tail, structure, distribution, and runtime metrics
- `report_generator.py`: Markdown report generation

## Dependencies

- pandas >= 1.5.0
- numpy >= 1.23.0
- scipy >= 1.9.0
- pyarrow >= 10.0.0

## Performance

- **Parallel Processing**: The module uses multiprocessing to process portfolios in parallel
  - Default: Uses all available CPU cores
  - Can be controlled with `--n-jobs` parameter
  - For 100,000 portfolios with 2 confidence levels × 2 horizons = 400,000 combinations
  - Expected speedup: ~N× where N is the number of CPU cores

## Notes

- The module processes **all portfolios** by default (no sampling limit)
- All paths are resolved relative to `implementation_03` root directory
- VaR violations occur when actual return < -VaR
- Traffic light zones follow Basel guidelines (adjusted for confidence level)
- Progress reporting shows completion rate, processing speed, and estimated time remaining

