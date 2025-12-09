# Research Methodology: Quantum vs Classical Risk Assessment and Portfolio Optimization

## 1. Introduction

This document presents a comprehensive research methodology for comparative analysis of quantum computing algorithms versus classical methods in financial risk assessment (Value-at-Risk and Conditional Value-at-Risk estimation) and portfolio optimization. The research framework is designed to provide rigorous empirical evaluation of quantum advantage claims through systematic benchmarking, scalability analysis, and statistical validation.

### 1.1 Research Objectives

#### Primary Objective 1: Quantum vs Classical Risk Assessment
**Research Question:** Can quantum amplitude estimation (QAE) or quantum Monte Carlo methods estimate portfolio tail risk (VaR/CVaR) faster or more accurately than classical Monte Carlo simulation?

**Hypotheses:**
- **H₁**: QAE demonstrates superior convergence rate (O(1/N) vs O(1/√N)) compared to classical Monte Carlo for CVaR estimation
- **H₂**: Quantum methods achieve comparable accuracy to classical methods with significantly fewer samples
- **H₃**: Hybrid workflows (classical scenario generation + quantum evaluation) offer computational advantages

#### Primary Objective 2: Quantum-Enhanced Portfolio Optimization
**Research Question:** Do Quantum Mean-Variance (QMV) or QAOA-based CVaR optimization achieve superior solution quality, runtime efficiency, or scalability compared to classical optimization methods?

**Hypotheses:**
- **H₄**: Quantum optimization methods produce portfolios with superior risk-adjusted returns (Sharpe ratio) under constraints
- **H₅**: QUBO formulations enable efficient scaling to larger asset universes (N > 30) relative to classical solvers
- **H₆**: Quantum methods yield sparser portfolio solutions with better interpretability

### 1.2 Scope and Limitations

**Dataset:**
- 10 years of daily OHLCV data for 10 stocks
- Time period: Historical daily data with sufficient coverage for rolling window analysis
- Data quality: Handles missing values, alignment, and outlier treatment

**Methodological Limitations:**
- Quantum methods implemented via classical simulation (Qiskit Aer) pending hardware access
- Portfolio size limited by computational resources and available data
- Backtesting horizon constrained by data availability

---

## 2. Data Preparation and Preprocessing

### 2.1 Data Loading

**Source Format:**
- CSV files: `{TICKER}_10yr_daily.csv`
- Structure:
  - Row 1: Column headers (Price, Close, High, Low, Open, Volume)
  - Row 2: Ticker information
  - Row 3: Date label
  - Row 4+: Time series data (Date, OHLCV)

**Processing Pipeline:**
1. **Symbol Extraction**: Identify available tickers from filenames
2. **Multi-Stock Loading**: Parallel loading of N stocks (default: 10)
3. **Data Validation**: Check for required columns, date format consistency

### 2.2 Data Cleaning

**Cleaning Operations:**

1. **Missing Value Treatment:**
   - Forward fill for gaps within trading days
   - Interpolation for isolated missing values
   - Row removal for excessive missing data (>10% per stock)

2. **Temporal Alignment:**
   - Align all stocks to common date index
   - Handle market holidays and non-trading days
   - Ensure consistent date ordering (ascending)

3. **Outlier Detection and Treatment:**
   - Identify extreme returns (>5σ from mean)
   - Winsorization at 1st and 99th percentiles
   - Verify price continuity (max daily change <50%)

4. **Data Quality Metrics:**
   - Missing data percentage per stock
   - Coverage statistics (dates with all stocks)
   - Final panel data dimensions: `(T × N)` where T = trading days, N = assets

### 2.3 Feature Engineering

**Returns Calculation:**

1. **Log Returns (Primary):**
   ```
   r_{t,i} = ln(P_{t,i} / P_{t-1,i})
   ```
   where:
   - `r_{t,i}`: log return of asset `i` on day `t`
   - `P_{t,i}`: adjusted close price of asset `i` on day `t`

2. **Simple Returns (Alternative):**
   ```
   R_{t,i} = (P_{t,i} / P_{t-1,i}) - 1
   ```

**Statistical Moments:**

1. **Mean Returns (Annualized):**
   ```
   μ_i = 252 × E[r_{t,i}]
   ```
   Vector: `μ = [μ_1, μ_2, ..., μ_N]^T`

2. **Covariance Matrix (Annualized):**
   ```
   Σ_{i,j} = 252 × Cov(r_{t,i}, r_{t,j})
   ```
   Matrix: `Σ ∈ ℝ^{N×N}`

3. **Volatility (Annualized):**
   ```
   σ_i = √(252 × Var(r_{t,i}))
   ```

**Validation Checks:**
- Positive definiteness of covariance matrix
- Condition number < 10⁶
- Stationarity tests (Augmented Dickey-Fuller)

---

## 3. Portfolio Generation Methodology

### 3.1 Dirichlet Distribution Sampling

**Rationale:**
The Dirichlet distribution ensures portfolio weights sum to unity while allowing for diverse allocation patterns, making it suitable for generating realistic portfolio scenarios.

**Mathematical Formulation:**

A random portfolio weight vector `w = [w₁, w₂, ..., wₙ]^T` is sampled from:

```
w ~ Dirichlet(α₁, α₂, ..., αₙ)
```

where `α_i > 0` are concentration parameters.

**Properties:**
- Constraint satisfaction: `∑ᵢ wᵢ = 1`, `wᵢ ≥ 0` (long-only)
- Distribution shape controlled by `α`:
  - `αᵢ = 1` ∀i: Uniform distribution
  - `αᵢ < 1`: Sparse/concentrated portfolios
  - `αᵢ > 1`: Diversified portfolios

**Implementation:**
```python
from numpy.random import dirichlet
weights = dirichlet(alpha=[1.0] * num_assets, size=num_portfolios)
```

**Generation Parameters:**
- Total portfolios: 100,000
- Concentration parameter: `α = [1.0, ..., 1.0]` (uniform prior)
- Additional constraints: None (unconstrained generation)

### 3.2 Portfolio Diversity

**Target Distribution:**
- Equal-weight portfolios (10% allocation each)
- Concentrated portfolios (single asset >50%)
- Diversified portfolios (weights spread across all assets)
- Sector-concentrated patterns (when applicable)

**Quality Metrics:**
- Effective number of assets: `N_eff = 1 / ∑ᵢ wᵢ²`
- Maximum weight: `w_max = max(w)`
- Weight entropy: `H(w) = -∑ᵢ wᵢ ln(wᵢ)`

---

## 4. Classical Risk Assessment Methods

### 4.1 Parametric VaR (Variance-Covariance Method)

**Mathematical Foundation:**

The Parametric VaR assumes portfolio returns follow a multivariate normal distribution:

```
R_p = w^T R ~ N(μ_p, σ_p²)
```

where:
- `R_p`: Portfolio return
- `w`: Portfolio weights
- `R`: Asset returns vector
- `μ_p = w^T μ`: Portfolio expected return
- `σ_p² = w^T Σ w`: Portfolio variance

**VaR Calculation:**

For confidence level `α` (e.g., 95% → α = 0.05):

```
VaR_{α}(T) = -[μ_p × (T/252) - z_{1-α} × σ_p × √(T/252)]
```

where:
- `T`: Risk horizon (days)
- `z_{1-α}`: Standard normal quantile (e.g., z₀.₉₅ = 1.645)

**Alternative (Zero-mean assumption):**
```
VaR_{α}(T) = -z_{1-α} × σ_p × √(T/252)
```

**Implementation Details:**
- Annualization factor: 252 trading days
- Confidence levels: 95%, 99%
- Risk horizon: 1 day (default)

**Advantages:**
- Computational efficiency: O(N²) for covariance matrix
- No simulation required
- Closed-form solution

**Limitations:**
- Assumes normal distribution (underestimates tail risk)
- Fails during market stress (non-normal tails)
- No CVaR estimation directly

### 4.2 Monte Carlo VaR/CVaR

**Methodology:**

Monte Carlo simulation generates synthetic return scenarios from the estimated return distribution.

**Algorithm:**

1. **Distribution Estimation:**
   - Estimate `μ` and `Σ` from historical returns
   - Annualize: `μ_annual = 252 × μ`, `Σ_annual = 252 × Σ`

2. **Scenario Generation:**
   ```
   R^{(s)} ~ N(μ_annual, Σ_annual), s = 1, ..., S
   ```
   where `S` is the number of Monte Carlo paths (typically 10⁴–10⁶)

3. **Portfolio Return Simulation:**
   ```
   R_p^{(s)} = w^T R^{(s)}
   ```

4. **VaR Estimation:**
   ```
   VaR_{α} = -Percentile({R_p^{(s)}}, α × 100)
   ```

5. **CVaR (Expected Shortfall) Estimation:**
   ```
   CVaR_{α} = -E[R_p^{(s)} | R_p^{(s)} ≤ -VaR_{α}]
   ```

**Sample Sizes Tested:**
- S ∈ {1,000, 5,000, 10,000, 50,000, 100,000}

**Convergence Analysis:**
- Classical MC error: `ε_MC ∝ 1/√S` (O(1/√N) convergence)
- Root Mean Square Error (RMSE) vs. analytical baseline
- Confidence interval estimation via bootstrap

**Implementation:**
- Random seed: 42 (reproducibility)
- Multivariate normal sampling via `numpy.random.multivariate_normal`
- Empirical quantile estimation

**Advantages:**
- Handles non-normal distributions (via copulas)
- Provides both VaR and CVaR
- Flexible distributional assumptions

**Disadvantages:**
- Computational cost: O(S × N²)
- Convergence rate: O(1/√S)
- Requires large sample sizes for tail accuracy

### 4.3 GARCH(1,1) Volatility Forecasting

**Model Specification:**

Generalized Autoregressive Conditional Heteroskedasticity models capture time-varying volatility:

**GARCH(1,1) Process:**
```
σ_t² = ω + α × ε_{t-1}² + β × σ_{t-1}²
```

where:
- `ω > 0`: Long-run variance
- `α ≥ 0`: ARCH coefficient (shock impact)
- `β ≥ 0`: GARCH coefficient (volatility persistence)
- Constraint: `α + β < 1` (stationarity)

**Forecasting:**
```
σ_{t+h|t}² = ω/(1-α-β) + (α+β)^h × [σ_t² - ω/(1-α-β)]
```

**Portfolio VaR with GARCH:**
```
VaR_{α,t} = -z_{1-α} × σ_{p,t} × √(T/252)
```

where `σ_{p,t}² = w^T Σ_t w` and `Σ_t` is the time-varying covariance matrix.

**Implementation:**
- Library: `arch` package (Python)
- Model: GARCH(1,1) per asset
- Forecasting horizon: 1-day ahead

**Use Case:**
- Dynamic risk estimation
- Time-varying volatility adjustment
- Stress testing scenarios

### 4.4 Extreme Value Theory (EVT) - Peaks Over Threshold

**Mathematical Foundation:**

EVT models the tail of loss distributions beyond a threshold `u` using the Generalized Pareto Distribution (GPD).

**POT Methodology:**

1. **Threshold Selection:**
   - Select threshold `u` (e.g., 90th or 95th percentile)
   - Extract exceedances: `Y_i = X_i - u` where `X_i > u`

2. **GPD Fit:**
   ```
   G_{ξ,σ}(y) = 1 - [1 + (ξy/σ)]^{-1/ξ}, ξ ≠ 0
   ```
   where:
   - `ξ`: Shape parameter (tail index)
   - `σ`: Scale parameter

3. **VaR Estimation:**
   ```
   VaR_{α} = u + (σ/ξ) × [(n/N_u × (1-α))^{-ξ} - 1]
   ```
   where:
   - `n`: Total observations
   - `N_u`: Number of exceedances

4. **CVaR Estimation:**
   ```
   CVaR_{α} = VaR_{α} + (σ + ξ(VaR_{α} - u))/(1 - ξ)
   ```

**Advantages:**
- Specifically models tail risk
- No distributional assumptions on body
- Suitable for extreme events

**Disadvantages:**
- Threshold selection sensitivity
- Requires sufficient tail data
- Parameter estimation uncertainty

**Implementation:**
- Threshold: 90th percentile of losses
- Maximum likelihood estimation (MLE) for GPD parameters
- Confidence levels: 95%, 99%

---

## 5. Quantum Risk Assessment Methods

### 5.1 Quantum Amplitude Estimation (QAE) for CVaR

**Theoretical Background:**

QAE provides quadratic speedup in estimating expected values of random variables compared to classical Monte Carlo.

**Classical vs Quantum Convergence:**
- Classical MC: `ε ∝ 1/√M` (requires M samples)
- Quantum QAE: `ε ∝ 1/M` (requires M oracle calls)

**CVaR Formulation as Amplitude Estimation:**

Given portfolio loss distribution `L(w, R)`, CVaR at confidence level `α`:

```
CVaR_α = E[L | L ≥ VaR_α]
```

**Quantum Encoding:**

1. **State Preparation:**
   ```
   |ψ⟩ = ∑_i √p_i |i⟩ |L_i⟩
   ```
   where:
   - `p_i`: Probability of scenario `i`
   - `L_i`: Loss value for scenario `i`

2. **Oracle Construction:**
   - Indicator function: `O|i⟩ = (-1)^{f(i)} |i⟩` where `f(i) = 1` if `L_i ≥ VaR_α`
   - Amplitude amplification to enhance tail states

3. **Estimation:**
   - QAE algorithm estimates amplitude of tail states
   - CVaR extracted from estimated amplitude

**Circuit Structure:**
- Qubits: `log₂(M)` for M scenarios (typically 4-12 qubits)
- Depth: `O(M)` with amplitude estimation
- Shots: 8,192 (default)

**Implementation Status:**
- Current: Classical approximation (Qiskit Aer simulator)
- Quantum circuit design: Placeholder structure
- Full implementation requires Qiskit Runtime or hardware access

**Hybrid Workflow:**
1. Classical scenario generation (Monte Carlo)
2. Quantum encoding of loss distribution
3. QAE for tail expectation estimation
4. Classical post-processing

### 5.2 Quantum Approximate Optimization Algorithm (QAOA) for CVaR Risk

**Problem Formulation:**

Minimize portfolio risk subject to CVaR constraints:

```
min_w w^T Σ w
s.t. CVaR_α(w) ≤ threshold
     ∑w_i = 1, w_i ≥ 0
```

**QUBO Transformation:**

Binary encoding of continuous weights:
```
w_i = ∑_{k=0}^{K-1} b_{i,k} / 2^k
```

QUBO form:
```
H = -∑_i ∑_j Q_{i,j} b_i b_j + ∑_i h_i b_i
```

where `Q` encodes covariance and CVaR constraints.

**QAOA Circuit:**
```
|β,γ⟩ = U_B(β_p) U_C(γ_p) ... U_B(β_1) U_C(γ_1) |+⟩^⊗n
```

where:
- `U_C(γ) = e^{-iγH_C}`: Problem Hamiltonian
- `U_B(β) = e^{-iβH_B}`: Mixer Hamiltonian
- `p`: Number of layers

**Optimization:**
- Classical optimizer (COBYLA, SPSA) for `β, γ`
- Expectation value: `⟨β,γ|H_C|β,γ⟩`

**Parameters:**
- Layers: `p = 2` (default)
- Shots: 8,192
- Classical optimizer: COBYLA

### 5.3 Quantum Generative Adversarial Network (QGAN) for Scenario Generation

**Concept:**

QGAN generates synthetic return scenarios using quantum circuits, potentially capturing complex return distributions.

**Architecture:**
- **Generator (G)**: Quantum variational circuit generating scenarios
- **Discriminator (D)**: Classical neural network distinguishing real vs. generated

**Loss Function:**
```
L_GAN = E[log D(R_real)] + E[log(1 - D(G(z)))]
```

**Use Case:**
- Synthetic scenario generation for stress testing
- Capturing non-normal return distributions
- Data augmentation for limited historical data

**Status:**
- Conceptual framework
- Implementation pending hardware access

### 5.4 Quantum Principal Component Analysis (qPCA) for Factor Risk

**Mathematical Foundation:**

qPCA extracts principal components of covariance matrix `Σ` using quantum algorithms.

**Quantum Algorithm:**
1. **Encoding:** `Σ → ρ` (density matrix)
2. **Phase Estimation:** Extract eigenvalues `λ_i`
3. **Projection:** Identify dominant risk factors

**Factor Model:**
```
R = F × Λ + ε
```

where:
- `F`: Factor loadings
- `Λ`: Factor returns
- `ε`: Idiosyncratic risk

**Risk Decomposition:**
```
σ_p² = w^T F × Cov(Λ) × F^T w + w^T diag(ε) w
```

**Advantages:**
- Dimensionality reduction (N factors << N assets)
- Interpretable risk factors
- Efficient for large N

**Status:**
- Theoretical framework
- Implementation via Qiskit Finance (if available)

---

## 6. Classical Portfolio Optimization Methods

### 6.1 Markowitz Mean-Variance Optimization

**Mathematical Formulation:**

Maximize risk-adjusted return:

```
max_w w^T μ - λ × w^T Σ w
s.t. ∑w_i = 1
     w_i ≥ 0 (long-only)
```

where `λ ≥ 0` is the risk aversion parameter.

**Efficient Frontier:**

Alternative formulation (minimize variance for target return):

```
min_w w^T Σ w
s.t. w^T μ ≥ μ_target
     ∑w_i = 1
     w_i ≥ 0
```

**Solution Methods:**
1. **Quadratic Programming (QP):**
   - Solver: CVXPY with ECOS/OSQP
   - Complexity: O(N³)

2. **Closed-Form (Unconstrained):**
   ```
   w* = (1/λ) × Σ^{-1} μ
   ```
   (Requires normalization and projection for constraints)

**Performance Metrics:**
- Sharpe Ratio: `SR = (w^T μ) / √(w^T Σ w)`
- Expected Return: `R_p = w^T μ`
- Volatility: `σ_p = √(w^T Σ w)`

**Implementation:**
- Risk aversion: `λ = 1.0` (default)
- Constraints: Long-only, fully invested
- Efficient frontier: 50 points

### 6.2 Black-Litterman Model

**Theoretical Background:**

Combines market equilibrium returns with investor views:

**Market Equilibrium Returns:**
```
Π = δ × Σ × w_market
```

where:
- `Π`: Implied equilibrium returns
- `δ`: Risk aversion parameter
- `w_market`: Market capitalization weights

**Posterior Returns:**
```
μ_BL = [(τΣ)^{-1} + P^T Ω^{-1} P]^{-1} × [(τΣ)^{-1} Π + P^T Ω^{-1} Q]
```

where:
- `P`: Pick matrix (views)
- `Q`: View returns vector
- `Ω`: Uncertainty matrix
- `τ`: Confidence scaling

**Optimization:**
- Use `μ_BL` in Markowitz optimization
- Yields more stable, realistic portfolios

**Implementation:**
- Views: None specified (uses equilibrium only)
- `τ = 0.05` (default)
- `Ω = diag(P × (τΣ) × P^T)`

### 6.3 Risk Parity / Equal Risk Contribution (ERC)

**Objective:**

Equalize risk contribution from each asset:

```
RC_i(w) = w_i × (∂σ_p / ∂w_i) = w_i × (Σw)_i / σ_p
```

**Optimization:**
```
min_w ∑_i ∑_j (RC_i(w) - RC_j(w))²
s.t. ∑w_i = 1
     w_i ≥ 0
```

**Solution:**
- Iterative optimization (non-convex)
- Alternating direction method of multipliers (ADMM)

**Properties:**
- Diversified risk exposure
- Less sensitive to expected return estimates
- Lower turnover than mean-variance

**Implementation:**
- Solver: Scipy optimization (SLSQP)
- Convergence tolerance: 10⁻⁶

### 6.4 CVaR Optimization

**Problem Formulation:**

Minimize CVaR subject to return constraint:

```
min_w CVaR_α(w)
s.t. w^T μ ≥ μ_min
     ∑w_i = 1
     w_i ≥ 0
```

**Linear Programming Reformulation:**

For scenario-based CVaR (S scenarios):

```
min_{w,ξ,η} ξ + (1/(αS)) × ∑_s η_s
s.t. -w^T R^{(s)} - ξ ≤ η_s, ∀s
     η_s ≥ 0, ∀s
     w^T μ ≥ μ_min
     ∑w_i = 1
     w_i ≥ 0
```

where:
- `ξ`: VaR estimate
- `η_s`: Excess loss over VaR in scenario `s`

**Solver:**
- Linear programming: CVXPY with ECOS
- Scenarios: 10,000 Monte Carlo paths

**Advantages:**
- Coherent risk measure
- Focuses on tail risk
- Handles non-normal distributions

---

## 7. Quantum Portfolio Optimization Methods

### 7.1 Quantum Mean-Variance (QMV) via QUBO

**QUBO Formulation:**

Convert Markowitz problem to Quadratic Unconstrained Binary Optimization:

**Binary Encoding:**
```
w_i = (1/2^K) × ∑_{k=0}^{K-1} 2^k × b_{i,k}
```

**QUBO Hamiltonian:**
```
H = -∑_i μ_i w_i + λ × ∑_i ∑_j Σ_{i,j} w_i w_j + A × (∑_i w_i - 1)²
```

where `A` is a penalty coefficient for the budget constraint.

**Quantum Solvers:**

1. **Quantum Annealing (D-Wave):**
   - Hardware: D-Wave quantum annealer
   - Maps QUBO to Ising model
   - Finds ground state

2. **QAOA:**
   - Variational quantum algorithm
   - Circuit-based execution
   - Hybrid quantum-classical optimization

**Solution Extraction:**
- Binary solution → Continuous weights
- Post-processing: Normalization, constraint satisfaction

**Performance:**
- Solution quality vs. classical QP
- Runtime scaling: O(2^{NK}) vs. O(N³)
- Approximation quality

**Implementation:**
- Classical QUBO solver (placeholder)
- Qiskit QAOA (simulator)
- Risk aversion: `λ = 1.0`

### 7.2 QAOA for CVaR Portfolio Optimization

**Problem Encoding:**

Minimize CVaR via QAOA:

**QUBO for CVaR:**
```
H = CVaR_α(w) + A₁ × (∑w_i - 1)² + A₂ × max(0, μ_min - w^T μ)
```

**QAOA Circuit:**
- Problem Hamiltonian: `H_C` (CVaR + constraints)
- Mixer: `H_B = ∑_i X_i`
- Layers: `p = 2`

**Optimization Loop:**
1. Initialize `β, γ` randomly
2. Execute QAOA circuit (shots: 8,192)
3. Measure expectation: `⟨H_C⟩`
4. Update `β, γ` via classical optimizer
5. Repeat until convergence

**Parameters:**
- Confidence level: `α = 0.95`
- Scenarios: 10,000 (classical generation)
- Optimizer: COBYLA

### 7.3 QAE for CVaR Portfolio Optimization

**Concept:**

Use QAE to estimate CVaR within optimization loop:

**Algorithm:**
1. For candidate portfolio `w`:
   - Generate loss scenarios
   - Encode in quantum state
   - Use QAE to estimate `CVaR_α(w)`
2. Optimize `w` to minimize QAE-estimated CVaR
3. Iterate

**Advantages:**
- Faster CVaR estimation
- Hybrid quantum-classical loop

**Status:**
- Theoretical framework
- Implementation pending

---

## 8. Evaluation Framework

### 8.1 Risk Metrics

**Value-at-Risk (VaR):**
- Definition: Maximum loss not exceeded with probability `α`
- Confidence levels: 95%, 99%
- Estimation error: `RMSE = √(1/M × ∑(VaR_est - VaR_true)²)`

**Conditional Value-at-Risk (CVaR):**
- Definition: Expected loss given loss exceeds VaR
- Estimation error: RMSE vs. analytical/computational baseline

**Coverage Tests:**
- VaR backtesting: Count violations `V = ∑_t 1_{L_t > VaR_α}`
- Expected violations: `E[V] = α × T`
- Kupiec test: `LR = -2 ln[(1-α)^{T-V} α^V] + 2 ln[(1-V/T)^{T-V} (V/T)^V]` ~ χ²(1)

### 8.2 Portfolio Performance Metrics

**Return Metrics:**
- Annualized Return: `R_annual = 252 × mean(r_p)`
- Excess Return: `R_excess = R_annual - R_f` (risk-free rate)

**Risk Metrics:**
- Annualized Volatility: `σ_annual = √252 × std(r_p)`
- Maximum Drawdown: `MDD = max_t max_{s≤t} (V_s - V_t)/V_s`
- Downside Deviation: `DD = √(252/T × ∑_{r<0} r²)`

**Risk-Adjusted Returns:**
- Sharpe Ratio: `SR = R_excess / σ_annual`
- Sortino Ratio: `S = R_excess / DD`
- Calmar Ratio: `CR = R_annual / MDD`

**Solution Quality:**
- Sparsity: `S = (∑1_{|w_i| > ε}) / N` (ε = 0.01)
- Concentration: `C = max(w_i)`
- Turnover: `TO = ∑|w_{t+1,i} - w_{t,i}|`

### 8.3 Computational Performance

**Runtime Metrics:**
- Wall-clock time per portfolio/method
- Scaling analysis: `T(N)` vs. `N` (assets)
- Quantum resource requirements:
  - Qubits: `n_qubits`
  - Circuit depth: `d`
  - Number of shots: `M`

**Convergence Analysis:**
- Error vs. sample size: `ε(S)` for MC, `ε(M)` for QAE
- Fitting: `ε = a × S^b` (estimate `b` for convergence rate)

### 8.4 Statistical Tests

**Method Comparison:**
1. **Paired t-test:** Compare Sharpe ratios across methods
2. **Mann-Whitney U-test:** Non-parametric comparison
3. **Bootstrap confidence intervals:** 95% CI for performance metrics

**Significance Level:**
- `α = 0.05` for all tests
- Multiple comparison correction (Bonferroni) when applicable

---

## 9. Experimental Design

### 9.1 Risk Assessment Experiments

**Experiment 1.1: Convergence Analysis**
- **Objective:** Compare QAE vs. Monte Carlo convergence
- **Design:**
  - Sample portfolios: 20 (randomly selected)
  - Monte Carlo sample sizes: {1K, 5K, 10K, 50K, 100K}
  - QAE shots: 8,192
  - Confidence levels: 95%, 99%
  - Metrics: VaR/CVaR estimates, RMSE, runtime

**Experiment 1.2: Method Comparison**
- **Methods:** Parametric VaR, Monte Carlo, QAE, EVT POT
- **Portfolios:** 20 diverse portfolios
- **Metrics:** VaR/CVaR estimates, coverage, runtime

**Experiment 1.3: Hybrid Workflow**
- **Objective:** Evaluate classical + quantum hybrid
- **Design:**
  - Classical scenario generation (MC)
  - Quantum evaluation (QAE)
  - Comparison: Full classical vs. hybrid

### 9.2 Portfolio Optimization Experiments

**Experiment 2.1: Method Comparison**
- **Methods:** Markowitz, Risk Parity, QMV, QAOA CVaR
- **Metrics:** Sharpe ratio, volatility, drawdown, sparsity, runtime
- **Backtesting:** Rolling windows (10 windows)

**Experiment 2.2: Scalability Analysis**
- **Asset counts:** N ∈ {10, 20, 30, 40, 50}
- **Methods:** Markowitz, QMV
- **Metrics:** Runtime `T(N)`, Sharpe ratio, solution quality

**Experiment 2.3: Constraint Sensitivity**
- **Constraints:** Long-only, sector limits, position limits
- **Methods:** All optimization methods
- **Metrics:** Feasibility, optimality gap

### 9.3 Backtesting Framework

**Rolling Window Design:**
- Training window: 3 years (756 trading days)
- Test horizon: 7 years (1,764 trading days)
- Rolling step: 1 day
- Minimum periods: 252 days

**Window Creation:**
```
Window 1: Train [t₀, t₀+755], Test [t₀+756, t₀+2519]
Window 2: Train [t₁, t₁+755], Test [t₁+756, t₁+2519]
...
Window K: Train [t_{K-1}, t_{K-1}+755], Test [t_{K-1}+756, t_{K-1}+2519]
```

**Re-balancing:**
- Frequency: 21 days (monthly)
- Transaction costs: Not included (can be added)

**Metrics Per Window:**
- Annualized return
- Volatility
- Sharpe ratio
- Maximum drawdown
- Calmar ratio

**Aggregation:**
- Mean across windows
- Standard deviation
- Worst-case (min Sharpe, max drawdown)

---

## 10. Results Structure and Reporting

### 10.1 Data Outputs

**Risk Comparison Data:**
- File: `results/comparative_analysis/risk_comparison.csv`
- Columns: method, sample_size, portfolio_id, confidence_level, var, cvar, runtime, num_qubits, circuit_depth
- Rows: ~320 (20 portfolios × 4 methods × 2 confidence levels × variable sample sizes)

**Portfolio Optimization Data:**
- File: `results/comparative_analysis/portfolio_optimization_comparison.csv`
- Columns: method, weights, expected_return, volatility, sharpe_ratio, runtime, sparsity, backtest_metrics
- Rows: Methods × windows

**Scalability Analysis:**
- File: `results/comparative_analysis/scalability_analysis.csv`
- Columns: method, num_assets, runtime, sharpe_ratio
- Rows: Methods × asset counts

**Backtest Results:**
- File: `results/backtest_results.csv`
- Columns: method, window_id, annualized_return, volatility, sharpe_ratio, max_drawdown, etc.
- Rows: Methods × windows

### 10.2 Visualizations

**Generated Figures:**
1. **Error vs. Sample Size:** Convergence comparison (MC vs. QAE)
2. **Runtime Comparison:** Execution time across methods
3. **Method Comparison Summary:** Performance metrics heatmap
4. **VaR/CVaR Comparison:** Estimates across methods
5. **Backtest Time Series:** Cumulative returns over time
6. **Portfolio Performance:** Risk-return scatter plots
7. **Scalability Analysis:** Runtime scaling with N

**Figure Generation Script:**
- `generate_visualizations.py`
- Formats: PNG, PDF (publication-ready)
- Style: Seaborn/Matplotlib professional styling

### 10.3 Statistical Analysis

**Comparative Analysis Script:**
- `generate_comparative_analysis.py`
- Generates all experimental data
- Performs statistical tests
- Produces summary statistics

**Summary Statistics:**
- File: `results/comparative_analysis/summary_statistics.json`
- Contents:
  - Mean/variance of metrics by method
  - Statistical test results
  - Confidence intervals

### 10.4 Reporting Standards

**Comparative Analysis Report:**
- `COMPARATIVE_ANALYSIS_REPORT.md`
- Sections:
  - Executive summary
  - Data availability assessment
  - Analysis capabilities
  - Recommendations

**Research Paper Structure (Suggested):**
1. Abstract
2. Introduction
3. Literature Review
4. Methodology (this document)
5. Experimental Setup
6. Results and Analysis
7. Discussion
8. Conclusion and Future Work

---

## 11. Implementation Details

### 11.1 Software Stack

**Core Dependencies:**
- NumPy ≥1.24.0: Numerical computation
- Pandas ≥2.0.0: Data manipulation
- SciPy ≥1.10.0: Statistical functions, optimization
- Scikit-learn ≥1.3.0: Machine learning utilities

**Financial Libraries:**
- Statsmodels ≥0.14.0: Econometric models (GARCH)
- CVXPY ≥1.3.0: Convex optimization
- Arch ≥5.0.0: GARCH modeling (if available)

**Quantum Libraries:**
- Qiskit ≥0.45.0: Quantum computing framework
- Qiskit Aer ≥0.13.0: Quantum simulators
- Qiskit Algorithms ≥0.2.0: Quantum algorithms
- Qiskit Finance ≥0.4.0: Quantum finance modules (if available)

**Visualization:**
- Matplotlib ≥3.7.0: Plotting
- Seaborn ≥0.12.0: Statistical visualization

**Other:**
- MLflow ≥2.5.0: Experiment tracking
- PyArrow ≥12.0.0: Parquet file support
- tqdm ≥4.65.0: Progress bars

### 11.2 Code Organization

**Module Structure:**
```
src/
├── config/              # Configuration management
├── data_loading/        # CSV data loading
├── data_cleaning/       # Data cleaning and alignment
├── feature_engineering/ # Returns calculation
├── portfolio_generation/ # Dirichlet portfolio sampling
├── classical_risk_engine/    # Classical risk methods
├── quantum_risk_engine/      # Quantum risk methods
├── classical_portfolio_engine/ # Classical optimization
├── quantum_portfolio_engine/   # Quantum optimization
├── evaluation_and_backtesting/ # Backtesting framework
└── experiment_tracking/       # MLflow integration
```

### 11.3 Experiment Tracking

**MLflow Integration:**
- Experiment name: `quantum_vs_classical_risk_portfolio`
- Logged parameters:
  - Number of assets
  - Method configurations
  - Sample sizes
- Logged metrics:
  - Performance metrics (Sharpe, return, volatility)
  - Runtime statistics
  - Accuracy metrics
- Artifacts:
  - Result CSV files
  - Generated portfolios
  - Configuration files

**Logging Directory:**
- `experiments/logs/`: Parameter and metric JSON files
- `mlruns/`: MLflow experiment runs

### 11.4 Reproducibility

**Random Seeds:**
- Portfolio generation: Seed 42
- Monte Carlo: Seed 42
- Quantum simulations: Seed 42 (if supported)

**Configuration:**
- Centralized config: `llm.json`
- Version control: Git
- Dependency locking: `requirements.txt`

---

## 12. Validation and Quality Assurance

### 12.1 Unit Tests

**Test Coverage:**
- Data loading: Validates file parsing
- Returns calculation: Compares with reference implementation
- Risk metrics: Validates against analytical solutions
- Optimization: Checks constraint satisfaction

### 12.2 Integration Tests

**Pipeline Validation:**
- End-to-end execution
- Data flow integrity
- Result consistency checks

### 12.3 Numerical Validation

**Benchmark Comparisons:**
- Parametric VaR: Compare with analytical formula
- Monte Carlo: Compare with known distributions
- Markowitz: Compare with CVXPY solutions
- Convergence tests: Verify error scaling

### 12.4 Sensitivity Analysis

**Parameter Sensitivity:**
- Risk aversion parameter (Markowitz)
- Sample sizes (Monte Carlo)
- QAOA layers
- Confidence levels

**Robustness:**
- Different market regimes
- Varying data quality
- Portfolio diversity

---

## 13. Limitations and Future Work

### 13.1 Current Limitations

1. **Quantum Hardware:**
   - Methods use classical simulation (Qiskit Aer)
   - Real quantum advantage requires hardware access
   - Noisy intermediate-scale quantum (NISQ) limitations

2. **Portfolio Size:**
   - Limited to 10 assets (data constraint)
   - Scalability analysis up to 50 assets (theoretical)

3. **Data Constraints:**
   - Single market (equity stocks)
   - Daily frequency only
   - Limited to 10 years of data

4. **Method Coverage:**
   - Some quantum methods in prototype stage
   - GARCH requires optional dependency
   - CVaR optimization requires CVXPY

### 13.2 Future Enhancements

1. **Hardware Integration:**
   - IBM Quantum Network access
   - D-Wave quantum annealer
   - Real quantum measurements

2. **Extended Dataset:**
   - Multiple asset classes (bonds, commodities)
   - Higher frequency data (intraday)
   - Extended time period

3. **Additional Methods:**
   - Quantum Machine Learning (QML) for return prediction
   - Quantum reinforcement learning for portfolio management
   - Variational Quantum Eigensolver (VQE) applications

4. **Advanced Analysis:**
   - Transaction cost modeling
   - Market impact analysis
   - Regime-dependent optimization

---

## 14. References and Citations

### 14.1 Classical Methods

1. **Markowitz (1952):** "Portfolio Selection," Journal of Finance
2. **Black & Litterman (1992):** "Global Portfolio Optimization," Financial Analysts Journal
3. **Rockafellar & Uryasev (2002):** "Conditional Value-at-Risk for General Loss Distributions," Journal of Banking & Finance

### 14.2 Quantum Methods

1. **Brassard et al. (2002):** "Quantum Amplitude Amplification and Estimation," Contemporary Mathematics
2. **Farhi et al. (2014):** "A Quantum Approximate Optimization Algorithm," arXiv:1411.4028
3. **Rebentrost et al. (2018):** "Quantum computational finance: Monte Carlo pricing of financial derivatives," Physical Review A

### 14.3 Risk Assessment

1. **J.P. Morgan (1996):** "RiskMetrics Technical Document"
2. **McNeil et al. (2015):** "Quantitative Risk Management: Concepts, Techniques and Tools," Princeton University Press

---

## 15. Appendix

### 15.1 Notation Glossary

**General:**
- `N`: Number of assets
- `T`: Number of time periods
- `w`: Portfolio weight vector (N×1)
- `μ`: Expected returns vector (N×1)
- `Σ`: Covariance matrix (N×N)
- `R`: Returns matrix (T×N)

**Risk Metrics:**
- `VaR_α`: Value-at-Risk at confidence level α
- `CVaR_α`: Conditional Value-at-Risk at confidence level α
- `σ_p`: Portfolio volatility
- `MDD`: Maximum Drawdown

**Quantum:**
- `|ψ⟩`: Quantum state vector
- `H`: Hamiltonian operator
- `n_q`: Number of qubits
- `M`: Number of quantum measurements/shots

### 15.2 Acronyms

- **QAE:** Quantum Amplitude Estimation
- **QAOA:** Quantum Approximate Optimization Algorithm
- **QUBO:** Quadratic Unconstrained Binary Optimization
- **QMV:** Quantum Mean-Variance
- **QGAN:** Quantum Generative Adversarial Network
- **qPCA:** Quantum Principal Component Analysis
- **VaR:** Value-at-Risk
- **CVaR:** Conditional Value-at-Risk
- **ERC:** Equal Risk Contribution
- **EVT:** Extreme Value Theory
- **GARCH:** Generalized Autoregressive Conditional Heteroskedasticity
- **POT:** Peaks Over Threshold
- **GPD:** Generalized Pareto Distribution

---

**Document Version:** 1.0  
**Last Updated:** December 2024  
**Author:** Research Implementation Team  
**Status:** Research Methodology - Publication Ready

