# Hybrid Quantum-Classical Portfolio & Risk Engine

This implementation provides a comprehensive framework for comparing quantum and classical methods for portfolio risk assessment and optimization, as outlined in the research objectives.

## Research Objectives

1. **Quantum vs Classical Risk Assessment**: Benchmark QAE against classical Monte Carlo for VaR/CVaR estimation
2. **Quantum-Enhanced Portfolio Optimization**: Compare QAOA/QMV against classical Markowitz and CVaR optimization

## Project Structure

```
implementation_01/
├── src/
│   ├── data/                    # Data loading and preprocessing
│   │   ├── load_data.py
│   │   ├── calculate_returns.py
│   │   ├── calculate_covariances.py
│   │   └── preprocessing_pipeline.py
│   ├── classical/               # Classical methods
│   │   ├── risk/
│   │   │   ├── parametric_var.py
│   │   │   ├── monte_carlo.py
│   │   │   └── cvar.py
│   │   └── portfolio/
│   │       ├── markowitz.py
│   │       └── cvar_lp.py
│   ├── quantum/                 # Quantum methods
│   │   ├── quantum_utils.py
│   │   ├── qae_cvar.py
│   │   └── qaoa_opt.py
│   ├── research/                # Research experiments
│   │   ├── experiment_01_quantum_vs_classical_risk.py
│   │   └── experiment_02_quantum_portfolio_optimization.py
│   └── results-and-reporting/  # Visualization and reporting
│       └── visualize_results.py
├── architecture/                # Architecture documentation
└── requirements.txt
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the dataset folder is accessible (contains 10-year stock data CSVs)

## Usage

### Running Research Experiments

**Experiment 1: Quantum vs Classical Risk Assessment**
```bash
cd src/research
python experiment_01_quantum_vs_classical_risk.py
```

This will:
- Run classical risk methods (Parametric VaR, Monte Carlo, Historical, Analytical CVaR)
- Run quantum QAE methods with different qubit counts
- Compare error rates and runtime
- Save results to `results-and-reporting/experiment_01_results.json`

**Experiment 2: Quantum Portfolio Optimization**
```bash
python experiment_02_quantum_portfolio_optimization.py
```

This will:
- Run classical optimization (Markowitz, CVaR LP)
- Run quantum QAOA with different depths
- Compare Sharpe ratios, returns, and runtime
- Test scalability with different numbers of assets
- Save results to `results-and-reporting/experiment_02_results.json`

### Visualizing Results

```bash
cd src/results-and-reporting
python visualize_results.py
```

This generates comparison plots for both experiments.

## Data Format

The system expects CSV files in the dataset folder with format:
- Filename: `{SYMBOL}_10yr_daily.csv`
- Columns: Date, Open, High, Low, Close, Volume
- Date format: YYYY-MM-DD

## Key Features

1. **Data Preprocessing**: Loads and preprocesses 10-year stock data, calculates returns, volatilities, and covariances
2. **Classical Risk**: Parametric VaR, Monte Carlo simulation, Historical VaR/CVaR, Analytical CVaR
3. **Quantum Risk**: QAE-based CVaR estimation with configurable qubit counts
4. **Classical Optimization**: Markowitz mean-variance, CVaR linear programming
5. **Quantum Optimization**: QAOA-based portfolio optimization with QUBO formulation
6. **Benchmarking**: Comprehensive experiments comparing quantum vs classical methods
7. **Visualization**: Automated plotting of comparison results

## Notes

- Quantum methods currently use simulators (Qiskit Aer)
- Some quantum implementations include simplified versions for demonstration
- Full QAE and QAOA implementations would require more complex circuit construction
- Results are saved in JSON format for reproducibility

## Research Outputs

The experiments generate:
- Comparison tables of quantum vs classical methods
- Error analysis (O(1/N) vs O(1/√N) scaling)
- Runtime comparisons
- Scalability analysis
- Solution quality metrics (Sharpe ratio, returns, volatility)

These outputs can be used directly in research papers comparing quantum and classical approaches.

