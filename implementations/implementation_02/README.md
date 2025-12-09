# Quantum vs Classical Risk and Portfolio Engine - Implementation 02

## Overview

This implementation provides a comprehensive comparative study of quantum vs classical methods for risk assessment (VaR/CVaR) and portfolio construction/optimization using 10 years of daily OHLCV data for 10 stocks.

## Project Structure

```
implementation_02/
├── src/
│   ├── config/                 # Configuration management
│   ├── data_loading/          # CSV data loading
│   ├── data_cleaning/          # Data cleaning and alignment
│   ├── feature_engineering/    # Returns calculation
│   ├── portfolio_generation/   # Dirichlet portfolio sampling
│   ├── classical_risk_engine/ # Classical risk methods (VaR, CVaR, GARCH, EVT)
│   ├── quantum_risk_engine/   # Quantum risk methods (QAE, QAOA, QGAN, qPCA)
│   ├── classical_portfolio_engine/ # Classical optimization (Markowitz, BL, Risk Parity, CVaR)
│   ├── quantum_portfolio_engine/   # Quantum optimization (QMV, QAOA, QAE)
│   ├── evaluation_and_backtesting/ # Rolling window backtesting
│   └── experiment_tracking/    # MLflow integration
├── llm.json                    # Project configuration
├── requirements.txt            # Python dependencies
└── run_pipeline.py            # Main execution script
```

## Features

### Classical Risk Methods
- **Parametric VaR**: Variance-covariance method
- **Monte Carlo VaR/CVaR**: Simulation-based risk estimation
- **GARCH Volatility**: GARCH(1,1) volatility forecasting
- **EVT POT**: Extreme Value Theory using Peaks Over Threshold

### Quantum Risk Methods
- **QAE CVaR**: Quantum Amplitude Estimation for CVaR
- **QAOA CVaR Risk**: QAOA for CVaR-based risk minimization
- **QGAN Scenario Generation**: Quantum GAN for synthetic scenario generation
- **qPCA Factor Risk**: Quantum PCA for factor risk analysis

### Classical Portfolio Methods
- **Markowitz Mean-Variance**: Traditional mean-variance optimization
- **Black-Litterman**: BL model with market views
- **Risk Parity/ERC**: Equal Risk Contribution portfolios
- **CVaR Optimization**: CVaR-based portfolio optimization

### Quantum Portfolio Methods
- **QMV/QUBO**: Quantum Mean-Variance using QUBO formulation
- **QAOA CVaR Portfolio**: QAOA for CVaR-based portfolio optimization
- **QAE CVaR Portfolio**: QAE for CVaR-based portfolio optimization

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Optional dependencies for full functionality:
```bash
pip install arch qiskit qiskit-aer qiskit-algorithms cvxpy mlflow
```

## Usage

Run the complete pipeline:
```bash
python run_pipeline.py
```

The pipeline will:
1. Load data from the dataset folder
2. Clean and align data
3. Calculate returns and statistics
4. Generate 100,000 random portfolios using Dirichlet distribution
5. Run risk analysis on sample portfolios
6. Optimize portfolios using various methods
7. Perform rolling window backtesting
8. Log results and metrics

## Results

Results are saved to:
- `results/`: CSV files with backtest results and portfolios
- `experiments/logs/`: Experiment logs and metrics
- `data/generated/`: Generated portfolios in Parquet format

## Configuration

The project configuration is defined in `llm.json`, including:
- Data configuration (source format, cleaning pipeline)
- Portfolio generation parameters
- Backtesting strategy (rolling windows)
- Risk and portfolio method configurations
- Evaluation metrics

## Notes

- Quantum methods use classical approximations when Qiskit is not available
- Some optimization methods require cvxpy for full functionality
- The pipeline is designed to work with or without optional dependencies
- Results are logged both locally and to MLflow (if available)

## Dataset

The dataset should contain CSV files named `{TICKER}_10yr_daily.csv` in the `Code-Space/dataset/` folder with the following structure:
- First row: Column headers (Price, Close, High, Low, Open, Volume)
- Second row: Ticker information
- Third row: Date label
- Subsequent rows: Date and OHLCV data

## Research Output

This implementation provides:
- Comparative analysis of quantum vs classical risk methods
- Portfolio optimization comparison
- Backtesting results across rolling windows
- Performance metrics (Sharpe ratio, volatility, drawdown, etc.)
- Runtime comparisons
- Risk metric accuracy (VaR/CVaR coverage, MAE, RMSE)

## Documentation

### Research Methodology
See **[METHODOLOGY.md](METHODOLOGY.md)** for a comprehensive research-level methodology document covering:
- Research objectives and hypotheses
- Mathematical formulations for all methods
- Experimental design and evaluation framework
- Statistical analysis approach
- Results structure and reporting standards

### Implementation Details
See **[IMPLEMENTATION.md](IMPLEMENTATION.md)** for detailed technical documentation including:
- System architecture and module structure
- Code-level implementation details
- Algorithm specifications
- Performance considerations
- Error handling and robustness

### Comparative Analysis
See **[COMPARATIVE_ANALYSIS_REPORT.md](COMPARATIVE_ANALYSIS_REPORT.md)** for data availability assessment and analysis capabilities.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{quantum_classical_risk_portfolio_2024,
  title={Quantum vs Classical Risk Assessment and Portfolio Optimization Framework},
  author={Research Implementation Team},
  year={2024},
  url={https://github.com/your-repo/implementation_02}
}
```

