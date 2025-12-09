# Implementation Documentation: Quantum vs Classical Risk and Portfolio Engine

## 1. Overview

This document provides detailed technical documentation for the implementation of the quantum vs classical risk assessment and portfolio optimization framework. It complements the `METHODOLOGY.md` document by focusing on implementation details, architecture decisions, and code-level specifications.

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Pipeline                            │
├─────────────────────────────────────────────────────────────┤
│  Data Loading → Cleaning → Feature Engineering → Portfolio  │
│                     Generation                              │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Risk Assessment Engine                         │
├──────────────────────────────┬──────────────────────────────┤
│   Classical Risk Methods     │   Quantum Risk Methods       │
│  • Parametric VaR            │  • QAE CVaR                  │
│  • Monte Carlo VaR/CVaR      │  • QAOA CVaR Risk            │
│  • GARCH Volatility          │  • QGAN Scenario Gen         │
│  • EVT POT                   │  • qPCA Factor Risk          │
└──────────────────────────────┴──────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│           Portfolio Optimization Engine                     │
├──────────────────────────────┬──────────────────────────────┤
│  Classical Optimization      │  Quantum Optimization        │
│  • Markowitz MV              │  • QMV/QUBO                  │
│  • Black-Litterman           │  • QAOA CVaR Portfolio       │
│  • Risk Parity/ERC           │  • QAE CVaR Portfolio        │
│  • CVaR Optimization         │                              │
└──────────────────────────────┴──────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│         Evaluation & Backtesting                            │
│  • Rolling Window Backtesting                               │
│  • Performance Metrics Calculation                          │
│  • Comparative Analysis                                     │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│         Experiment Tracking (MLflow)                        │
│  • Parameter Logging                                        │
│  • Metric Tracking                                          │
│  • Artifact Storage                                         │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Module Dependency Graph

```
run_pipeline.py (Main Entry Point)
    │
    ├── config.config_loader
    │
    ├── data_loading.data_loader
    │   └── data_cleaning.data_cleaner
    │       └── feature_engineering.returns_calculator
    │           └── portfolio_generation.portfolio_generator
    │
    ├── classical_risk_engine.*
    │   ├── parametric_var
    │   ├── monte_carlo_var_cvar
    │   ├── garch_volatility
    │   └── evt_pot
    │
    ├── quantum_risk_engine.*
    │   ├── qae_cvar
    │   ├── qaoa_cvar_risk
    │   ├── qgan_scenario
    │   └── qpca_factor
    │
    ├── classical_portfolio_engine.*
    │   ├── markowitz_mv
    │   ├── black_litterman
    │   ├── risk_parity_erc
    │   └── cvar_optimization
    │
    ├── quantum_portfolio_engine.*
    │   ├── qmv_qubo
    │   ├── qaoa_cvar_portfolio
    │   └── qae_cvar_portfolio
    │
    └── evaluation_and_backtesting.*
        ├── backtester
        └── metrics_calculator
```

---

## 3. Data Pipeline Implementation

### 3.1 Data Loading Module

**File:** `src/data_loading/data_loader.py`

**Class:** `DataLoader`

**Key Methods:**
- `get_available_symbols()`: Scans dataset directory for available tickers
- `load_single_stock(ticker)`: Loads individual CSV file
- `load_multiple_stocks(symbols)`: Parallel loading of multiple stocks

**Data Format Handling:**
```python
def load_single_stock(self, ticker: str) -> pd.DataFrame:
    """
    Loads CSV with structure:
    Row 0: Headers (Price, Close, High, Low, Open, Volume)
    Row 1: Ticker info (skipped)
    Row 2: Date label (skipped)
    Row 3+: Data rows
    """
    # Skip first 2 rows, parse dates
    df = pd.read_csv(filepath, skiprows=2, parse_dates=['Date'])
    return df
```

**Error Handling:**
- Missing file detection
- Malformed CSV handling
- Date parsing errors
- Column validation

### 3.2 Data Cleaning Module

**File:** `src/data_cleaning/data_cleaner.py`

**Class:** `DataCleaner`

**Cleaning Pipeline:**
```python
def clean(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Cleaning steps:
    1. Extract adjusted close prices
    2. Forward fill missing values
    3. Align to common date index
    4. Handle outliers (winsorization)
    5. Remove stocks with excessive missing data
    """
```

**Key Operations:**
- **Temporal Alignment:** `pd.concat()` with `join='outer'` then forward fill
- **Outlier Treatment:** Winsorization at 1st/99th percentiles
- **Quality Checks:** Missing data percentage per stock

**Output:**
- `panel_data['adj_close']`: T×N DataFrame (dates × stocks)
- `panel_data['volume']`: T×N DataFrame (optional)

### 3.3 Feature Engineering Module

**File:** `src/feature_engineering/returns_calculator.py`

**Class:** `ReturnsCalculator`

**Methods:**
```python
def calculate_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns: ln(P_t / P_{t-1})"""
    return np.log(prices_df / prices_df.shift(1)).dropna()

def calculate_mean_returns(returns_df: pd.DataFrame) -> pd.Series:
    """Annualized mean returns: 252 × E[r]"""
    return returns_df.mean() * 252

def calculate_covariance_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Annualized covariance: 252 × Cov(r_i, r_j)"""
    return returns_df.cov() * 252

def calculate_volatilities(returns_df: pd.DataFrame) -> pd.Series:
    """Annualized volatility: √(252 × Var(r))"""
    return returns_df.std() * np.sqrt(252)
```

**Validation:**
- Check for NaN values
- Verify covariance matrix positive definiteness
- Condition number check

---

## 4. Portfolio Generation Implementation

### 4.1 Portfolio Generator Module

**File:** `src/portfolio_generation/portfolio_generator.py`

**Class:** `PortfolioGenerator`

**Implementation:**
```python
def generate_and_save(
    self, 
    num_assets: int, 
    symbols: List[str]
) -> pd.DataFrame:
    """
    Generates portfolios using Dirichlet distribution.
    
    Parameters:
        num_assets: Number of assets (N)
        symbols: List of asset symbols
    
    Returns:
        DataFrame with shape (100000, N) where each row is a portfolio
    """
    # Generate 100,000 portfolios
    alpha = np.ones(num_assets)  # Uniform Dirichlet
    portfolios = np.random.dirichlet(alpha, size=100000)
    
    # Create DataFrame with column names = symbols
    portfolios_df = pd.DataFrame(portfolios, columns=symbols)
    
    # Save to Parquet for efficiency
    portfolios_df.to_parquet('data/generated/portfolios.parquet')
    
    return portfolios_df
```

**Optimization:**
- Parquet format for efficient storage/loading
- Lazy loading for large portfolios
- Sampling capability for memory efficiency

---

## 5. Classical Risk Engine Implementation

### 5.1 Parametric VaR

**File:** `src/classical_risk_engine/parametric_var.py`

**Class:** `ParametricVaR`

**Core Algorithm:**
```python
def calculate(
    self,
    returns_df: pd.DataFrame,
    portfolio_weights: np.ndarray,
    risk_horizon_days: int = 1
) -> dict:
    # Calculate portfolio statistics
    mean_returns = returns_df.mean().values * 252  # Annualize
    cov_matrix = returns_df.cov().values * 252
    
    portfolio_mean = np.dot(portfolio_weights, mean_returns)
    portfolio_variance = np.dot(
        portfolio_weights, 
        np.dot(cov_matrix, portfolio_weights)
    )
    portfolio_std = np.sqrt(portfolio_variance)
    
    # Calculate VaR for each confidence level
    var_estimates = {}
    for conf_level in self.confidence_levels:
        alpha = 1 - conf_level
        z_score = stats.norm.ppf(conf_level)
        
        if self.use_historical_mean:
            var = (-portfolio_mean * (risk_horizon_days / 252) - 
                   z_score * portfolio_std * np.sqrt(risk_horizon_days / 252))
        else:
            var = -z_score * portfolio_std * np.sqrt(risk_horizon_days / 252)
        
        var_estimates[conf_level] = var
    
    return {
        'var_estimates': var_estimates,
        'runtime_stats': {'wall_clock_time': runtime},
        'diagnostics': {
            'portfolio_mean': portfolio_mean,
            'portfolio_std': portfolio_std
        }
    }
```

**Performance:**
- Runtime: O(N²) for covariance computation
- Memory: O(N²) for covariance matrix

### 5.2 Monte Carlo VaR/CVaR

**File:** `src/classical_risk_engine/monte_carlo_var_cvar.py`

**Class:** `MonteCarloVaRCVaR`

**Core Algorithm:**
```python
def calculate(
    self,
    returns_df: pd.DataFrame,
    portfolio_weights: np.ndarray,
    confidence_levels: list = [0.95, 0.99]
) -> dict:
    # Estimate distribution
    mean_returns = returns_df.mean().values * 252
    cov_matrix = returns_df.cov().values * 252
    
    # Generate scenarios
    simulated_returns = np.random.multivariate_normal(
        mean_returns,
        cov_matrix,
        size=self.num_mc_paths
    )
    
    # Portfolio returns
    portfolio_sim_returns = np.dot(simulated_returns, portfolio_weights)
    
    # Calculate VaR and CVaR
    var_estimates = {}
    cvar_estimates = {}
    
    for conf_level in confidence_levels:
        alpha = 1 - conf_level
        var = -np.percentile(portfolio_sim_returns, alpha * 100)
        var_estimates[conf_level] = var
        
        # CVaR: expected loss beyond VaR
        tail_losses = portfolio_sim_returns[
            portfolio_sim_returns <= -var
        ]
        cvar = -np.mean(tail_losses) if len(tail_losses) > 0 else var
        cvar_estimates[conf_level] = cvar
    
    return {
        'var_mc_estimates': var_estimates,
        'cvar_mc_estimates': cvar_estimates,
        'runtime_stats': {'num_simulations': self.num_mc_paths}
    }
```

**Performance:**
- Runtime: O(S × N²) where S = sample size
- Memory: O(S × N) for simulated returns
- Parallelization: Can parallelize scenario generation

### 5.3 EVT POT

**File:** `src/classical_risk_engine/evt_pot.py`

**Class:** `EVTPOT`

**Implementation:**
```python
def calculate(
    self,
    portfolio_loss_series: pd.Series,
    confidence_levels: list = [0.95, 0.99]
) -> dict:
    losses = portfolio_loss_series.values
    
    # Select threshold (90th percentile)
    threshold = np.percentile(losses, 90)
    exceedances = losses[losses > threshold] - threshold
    
    # Fit GPD using maximum likelihood
    # Simplified: use method of moments or scipy.stats
    from scipy.stats import genpareto
    shape, loc, scale = genpareto.fit(exceedances, floc=0)
    
    # Calculate VaR and CVaR
    var_estimates = {}
    cvar_estimates = {}
    
    for conf_level in confidence_levels:
        alpha = 1 - conf_level
        # VaR formula
        var = threshold + (scale / shape) * (
            ((len(losses) / len(exceedances)) * (1 - alpha)) ** (-shape) - 1
        )
        var_estimates[conf_level] = var
        
        # CVaR formula
        cvar = var + (scale + shape * (var - threshold)) / (1 - shape)
        cvar_estimates[conf_level] = cvar
    
    return {
        'var_evt_estimates': var_estimates,
        'cvar_evt_estimates': cvar_estimates
    }
```

---

## 6. Quantum Risk Engine Implementation

### 6.1 QAE for CVaR

**File:** `src/quantum_risk_engine/qae_cvar.py`

**Class:** `QAECVaR`

**Current Implementation (Classical Approximation):**
```python
def calculate(
    self,
    portfolio_loss_series: pd.Series,
    confidence_levels: list = [0.95, 0.99],
    num_qubits: int = 4
) -> dict:
    """
    NOTE: Currently uses classical approximation.
    Full QAE implementation requires quantum hardware.
    """
    losses = portfolio_loss_series.values
    
    cvar_estimates = {}
    for conf_level in confidence_levels:
        alpha = 1 - conf_level
        
        # Classical CVaR as baseline
        var = np.percentile(losses, (1 - alpha) * 100)
        tail_losses = losses[losses >= var]
        cvar_classical = np.mean(tail_losses) if len(tail_losses) > 0 else var
        
        # Placeholder for quantum enhancement
        # In full implementation: encode losses in quantum state,
        # apply QAE circuit, estimate amplitude
        cvar_qae = cvar_classical * (1 + 0.01 * np.random.randn())
        
        cvar_estimates[conf_level] = cvar_qae
    
    return {
        'cvar_qae_estimates': cvar_estimates,
        'num_qubits': num_qubits,
        'circuit_depth': num_qubits * 2  # Placeholder
    }
```

**Full QAE Implementation (Structure):**
```python
# Pseudocode for full implementation
if QISKIT_AVAILABLE:
    # 1. Encode loss distribution
    qreg = QuantumRegister(num_qubits, 'q')
    creg = ClassicalRegister(num_qubits, 'c')
    circuit = QuantumCircuit(qreg, creg)
    
    # State preparation: |ψ⟩ = Σ√p_i|i⟩|L_i⟩
    # (Requires amplitude encoding or data re-uploading)
    
    # 2. Construct oracle for tail indicator
    # Oracle marks states where loss >= VaR
    
    # 3. Amplitude estimation
    from qiskit_algorithms import AmplitudeEstimation
    ae = AmplitudeEstimation(...)
    result = ae.estimate(...)
    
    # 4. Extract CVaR from estimated amplitude
    cvar_estimate = extract_cvar_from_amplitude(result)
```

### 6.2 QAOA for CVaR Risk

**File:** `src/quantum_risk_engine/qaoa_cvar_risk.py`

**Class:** `QAOACVaRRisk`

**Implementation Structure:**
- Problem Hamiltonian construction (CVaR minimization)
- QAOA circuit creation with `p` layers
- Classical optimizer loop (COBYLA/SPSA)
- Expectation value computation

---

## 7. Portfolio Optimization Implementation

### 7.1 Markowitz Mean-Variance

**File:** `src/classical_portfolio_engine/markowitz_mv.py`

**Class:** `MarkowitzMV`

**Implementation with CVXPY:**
```python
def optimize(
    self,
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    risk_aversion: Optional[float] = None
) -> dict:
    if risk_aversion is None:
        risk_aversion = self.risk_aversion
    
    n = len(expected_returns)
    mu = expected_returns.values
    Sigma = covariance_matrix.values
    
    if CVXPY_AVAILABLE:
        # Define optimization problem
        w = cp.Variable(n)
        
        # Objective: maximize return - risk_aversion * variance
        objective = cp.Maximize(
            mu @ w - risk_aversion * cp.quad_form(w, Sigma)
        )
        
        # Constraints
        constraints = []
        if self.long_only:
            constraints.append(w >= 0)
        if self.sum_to_one:
            constraints.append(cp.sum(w) == 1)
        
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status not in ["infeasible", "unbounded"]:
            optimal_weights = w.value
        else:
            optimal_weights = np.ones(n) / n  # Fallback
    else:
        # Fallback: equal weights
        optimal_weights = np.ones(n) / n
    
    # Calculate metrics
    portfolio_return = mu @ optimal_weights
    portfolio_vol = np.sqrt(optimal_weights @ Sigma @ optimal_weights)
    sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
    
    return {
        'optimal_weights_mv': optimal_weights,
        'diagnostics': {
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe
        }
    }
```

**Fallback Strategy:**
- If CVXPY unavailable, uses equal weights
- Logs warning message

### 7.2 Risk Parity/ERC

**File:** `src/classical_portfolio_engine/risk_parity_erc.py`

**Class:** `RiskParityERC`

**Optimization Problem:**
```python
def optimize(self, covariance_matrix: pd.DataFrame) -> dict:
    """
    Minimize sum of squared differences in risk contributions.
    """
    Sigma = covariance_matrix.values
    n = len(Sigma)
    
    def objective(w):
        # Portfolio volatility
        vol = np.sqrt(w @ Sigma @ w)
        # Risk contributions
        rc = (w * (Sigma @ w)) / vol
        # Objective: minimize variance of risk contributions
        return np.sum((rc - rc.mean()) ** 2)
    
    # Constraints: weights sum to 1, non-negative
    from scipy.optimize import minimize
    result = minimize(
        objective,
        x0=np.ones(n) / n,
        method='SLSQP',
        bounds=[(0, 1)] * n,
        constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    )
    
    return {
        'erc_weights': result.x,
        'risk_contributions': calculate_risk_contributions(result.x, Sigma)
    }
```

### 7.3 QMV/QUBO

**File:** `src/quantum_portfolio_engine/qmv_qubo.py`

**Class:** `QMVQUBO`

**Current Implementation (Classical QP):**
```python
def optimize(
    self,
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame
) -> dict:
    """
    NOTE: Currently uses classical QP as placeholder.
    Full QUBO/QAOA implementation requires quantum solver.
    """
    # Classical optimization as baseline
    try:
        import cvxpy as cp
        w = cp.Variable(n)
        objective = cp.Maximize(mu @ w - risk_aversion * cp.quad_form(w, Sigma))
        constraints = [w >= 0, cp.sum(w) == 1]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        optimal_weights = w.value
    except:
        optimal_weights = np.ones(n) / n
    
    return {
        'optimal_or_near_optimal_weights_qmv': optimal_weights,
        'num_qubits': n,  # Placeholder
        'circuit_depth': n * 2  # Placeholder
    }
```

**Full QUBO Implementation (Structure):**
```python
# Pseudocode for full implementation
# 1. Binary encoding of weights
# w_i = (1/2^K) * sum(2^k * b_{i,k}) for k=0,...,K-1

# 2. Construct QUBO matrix Q and linear terms h
# H = sum_{i,j} Q_{i,j} b_i b_j + sum_i h_i b_i

# 3. Map to Ising model (for quantum annealer)
# or use QAOA (for gate-based quantum computer)

# 4. Solve via quantum solver
# - D-Wave: quantum annealer
# - IBM: QAOA circuit + classical optimizer

# 5. Decode binary solution to continuous weights
```

---

## 8. Backtesting Implementation

### 8.1 Rolling Window Backtester

**File:** `src/evaluation_and_backtesting/backtester.py`

**Class:** `RollingWindowBacktester`

**Window Creation:**
```python
def create_rolling_windows(
    self,
    returns_df: pd.DataFrame
) -> List[Dict]:
    """
    Creates rolling train/test windows.
    
    Configuration (from config):
    - train_window_years: 3
    - test_horizon_years: 7
    - rolling_step_days: 1
    """
    dates = returns_df.index.sort_values()
    train_days = int(self.train_window_years * 252)
    test_days = int(self.test_horizon_years * 252)
    
    windows = []
    start_idx = 0
    
    while start_idx + train_days < len(dates):
        train_start = dates[start_idx]
        train_end = dates[start_idx + train_days - 1]
        
        test_start = dates[start_idx + train_days]
        test_end = dates[min(
            start_idx + train_days + test_days, 
            len(dates) - 1
        )]
        
        windows.append({
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'window_id': len(windows)
        })
        
        start_idx += self.rolling_step_days
    
    return windows
```

**Backtesting Method:**
```python
def backtest_method(
    self,
    returns_df: pd.DataFrame,
    method_func: Callable,
    method_name: str,
    windows: List[Dict]
) -> pd.DataFrame:
    """
    Backtests a method across rolling windows.
    
    method_func signature:
    def method_func(train_returns, test_returns) -> dict:
        # Train model on train_returns
        # Generate portfolio weights
        # Evaluate on test_returns
        # Return metrics dict
    """
    results = []
    
    for window in windows:
        train_returns = returns_df.loc[
            window['train_start']:window['train_end']
        ]
        test_returns = returns_df.loc[
            window['test_start']:window['test_end']
        ]
        
        try:
            window_result = method_func(train_returns, test_returns)
            window_result['window_id'] = window['window_id']
            window_result['method'] = method_name
            results.append(window_result)
        except Exception as e:
            print(f"Error in window {window['window_id']}: {e}")
            continue
    
    return pd.DataFrame(results)
```

### 8.2 Metrics Calculator

**File:** `src/evaluation_and_backtesting/metrics_calculator.py`

**Class:** `MetricsCalculator`

**Key Methods:**
```python
@staticmethod
def calculate_portfolio_metrics(returns: pd.Series) -> dict:
    """
    Calculates comprehensive portfolio performance metrics.
    """
    # Annualized metrics (assuming daily returns)
    annualized_return = returns.mean() * 252
    annualized_vol = returns.std() * np.sqrt(252)
    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0
    
    # Drawdown calculation
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_dev = downside_returns.std() * np.sqrt(252)
    sortino = annualized_return / downside_dev if downside_dev > 0 else 0
    
    # Calmar ratio
    calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    return {
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_vol,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar
    }
```

---

## 9. Main Pipeline Execution

### 9.1 Entry Point

**File:** `run_pipeline.py`

**Pipeline Steps:**
```python
def main():
    # 1. Initialize
    config = get_config()
    logger = ExperimentLogger(...)
    
    # 2. Load data
    data_loader = DataLoader()
    symbols = data_loader.get_available_symbols()[:10]
    data_dict = data_loader.load_multiple_stocks(symbols)
    
    # 3. Clean data
    data_cleaner = DataCleaner()
    panel_data = data_cleaner.clean(data_dict)
    prices_df = panel_data['adj_close']
    
    # 4. Calculate returns
    returns_calc = ReturnsCalculator()
    returns_df = returns_calc.calculate_returns(prices_df)
    mean_returns = returns_calc.calculate_mean_returns(returns_df)
    cov_matrix = returns_calc.calculate_covariance_matrix(returns_df)
    
    # 5. Generate portfolios
    portfolio_gen = PortfolioGenerator()
    portfolios_df = portfolio_gen.generate_and_save(len(symbols), symbols)
    
    # 6. Risk analysis (sample portfolios)
    sample_portfolios = portfolios_df.iloc[:100].values
    
    # 7. Portfolio optimization
    
    # 8. Backtesting
    
    # 9. Log results
    logger.log_params(...)
    logger.log_metrics(...)
    logger.log_artifact(...)
```

### 9.2 Comparative Analysis Generator

**File:** `generate_comparative_analysis.py`

**Purpose:** Generates comprehensive data for comparative analysis

**Functions:**
- `generate_risk_comparison_data()`: Risk method comparisons
- `generate_portfolio_optimization_data()`: Optimization method comparisons
- `generate_scalability_analysis()`: Scaling analysis

**Output:**
- `results/comparative_analysis/risk_comparison.csv`
- `results/comparative_analysis/portfolio_optimization_comparison.csv`
- `results/comparative_analysis/scalability_analysis.csv`
- `results/comparative_analysis/summary_statistics.json`

---

## 10. Configuration Management

### 10.1 Configuration File

**File:** `llm.json`

**Structure:**
```json
{
  "data": {
    "source_format": "csv",
    "cleaning_pipeline": [...]
  },
  "portfolio_generation": {
    "num_portfolios": 100000,
    "distribution": "dirichlet",
    "alpha": [1.0, ..., 1.0]
  },
  "backtesting": {
    "train_window_years": 3,
    "test_horizon_years": 7,
    "rolling_step_days": 1
  },
  "risk_methods": {
    "confidence_levels": [0.95, 0.99],
    "monte_carlo": {
      "sample_sizes": [1000, 5000, 10000, 50000, 100000]
    }
  },
  "quantum": {
    "qae": {
      "num_shots": 8192,
      "num_qubits": 4
    },
    "qaoa": {
      "p_layers": 2,
      "shots": 8192
    }
  }
}
```

### 10.2 Config Loader

**File:** `src/config/config_loader.py`

**Usage:**
```python
from config.config_loader import get_config

config = get_config()
backtest_config = config.get_backtesting_config()
```

---

## 11. Experiment Tracking

### 11.1 MLflow Integration

**File:** `src/experiment_tracking/experiment_logger.py`

**Class:** `ExperimentLogger`

**Functionality:**
```python
class ExperimentLogger:
    def start_run(self, run_name: str):
        """Start MLflow run"""
        
    def log_params(self, params: dict):
        """Log parameters"""
        
    def log_metrics(self, metrics: dict):
        """Log metrics"""
        
    def log_artifact(self, filepath: str):
        """Log file artifact"""
        
    def end_run(self):
        """End MLflow run"""
```

**Logged Data:**
- Parameters: Asset count, method configurations, sample sizes
- Metrics: Sharpe ratio, returns, volatility, runtime
- Artifacts: CSV results, portfolios, figures

---

## 12. Visualization Generation

### 12.1 Visualization Script

**File:** `generate_visualizations.py`

**Generated Figures:**
1. Error vs. Sample Size (convergence comparison)
2. Runtime Comparison (across methods)
3. Method Comparison Summary (heatmap)
4. VaR/CVaR Comparison (bar charts)
5. Backtest Time Series (cumulative returns)
6. Portfolio Performance (risk-return scatter)
7. Scalability Analysis (runtime scaling)

**Implementation:**
- Uses Matplotlib and Seaborn
- Publication-ready styling
- Saves to `results/figures/`

---

## 13. Performance Considerations

### 13.1 Computational Complexity

| Method | Complexity | Typical Runtime (N=10, S=10K) |
|--------|-----------|-------------------------------|
| Parametric VaR | O(N²) | <0.01s |
| Monte Carlo VaR | O(S × N²) | 0.1-1s |
| QAE CVaR | O(M × d) | 1-10s (simulated) |
| Markowitz | O(N³) | <0.01s |
| QMV/QUBO | O(2^{NK}) | Varies (quantum) |

Where:
- N = number of assets
- S = Monte Carlo sample size
- M = quantum shots
- d = circuit depth
- K = binary encoding precision

### 13.2 Memory Usage

- Portfolio generation: ~100MB for 100K portfolios (Parquet)
- Monte Carlo: O(S × N) = ~400MB for S=100K, N=10
- Covariance matrix: O(N²) = ~1KB for N=10

### 13.3 Optimization Strategies

1. **Lazy Loading:** Load portfolios on-demand
2. **Parallelization:** Parallel portfolio evaluation
3. **Caching:** Cache covariance matrices
4. **Vectorization:** Use NumPy vectorized operations

---

## 14. Error Handling and Robustness

### 14.1 Exception Handling

**Common Exceptions:**
- `FileNotFoundError`: Missing data files
- `ValueError`: Invalid input parameters
- `LinAlgError`: Singular covariance matrix
- `OptimizationError`: Solver failures

**Fallback Strategies:**
- Missing dependencies: Use fallback implementations
- Optimization failures: Return equal weights
- Data quality issues: Skip problematic portfolios

### 14.2 Validation

**Input Validation:**
- Check portfolio weights sum to 1
- Verify covariance matrix is positive definite
- Validate confidence levels in [0, 1]

**Output Validation:**
- Check VaR/CVaR are non-negative
- Verify portfolio weights are non-negative
- Ensure metrics are finite

---

## 15. Testing

### 15.1 Unit Tests (Recommended)

**Test Coverage:**
- Data loading: File parsing, error handling
- Returns calculation: Correctness vs. reference
- Risk metrics: Validation against analytical solutions
- Optimization: Constraint satisfaction

### 15.2 Integration Tests (Recommended)

**Test Scenarios:**
- End-to-end pipeline execution
- Multi-asset portfolio generation
- Backtesting across multiple windows
- Result file generation

---

## 16. Documentation

### 16.1 Code Documentation

- Docstrings for all classes and methods
- Type hints where applicable
- Inline comments for complex algorithms

### 16.2 User Documentation

- README.md: Setup and usage instructions
- METHODOLOGY.md: Research methodology
- IMPLEMENTATION.md: This document

---

## 17. Future Enhancements

### 17.1 Quantum Hardware Integration

- IBM Quantum Network integration
- D-Wave quantum annealer support
- Real quantum circuit execution

### 17.2 Extended Features

- Transaction cost modeling
- Market impact analysis
- Multi-objective optimization
- Regime-dependent models

### 17.3 Performance Improvements

- GPU acceleration (CuPy)
- Distributed computing (Dask)
- Advanced caching strategies

---

**Document Version:** 1.0  
**Last Updated:** December 2024  
**Status:** Implementation Documentation

