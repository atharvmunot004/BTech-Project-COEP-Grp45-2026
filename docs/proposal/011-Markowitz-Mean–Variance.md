## 1. Methodology for Implementation

### 1.1 Overview

Markowitz Mean-Variance Optimization is the foundational portfolio optimization framework that seeks to maximize expected return for a given level of risk (variance), or minimize risk for a given level of return. In your hedge fund system, this will serve as the **classical baseline optimizer** against which quantum and advanced classical methods (Black-Litterman, CVaR optimization, QAOA) are compared.

The Markowitz framework:
- Provides **efficient frontier** of optimal portfolios
- Balances risk-return tradeoff via quadratic optimization
- Forms the foundation for modern portfolio theory
- Serves as benchmark for quantum optimization (QMV, QAOA)

---

### 1.2 Inputs & How to Obtain Them

| Input | Type / Shape | Description | Source / Estimation |
|-------|-------------|-------------|-------------------|
| `mu` | vector (N,) | Expected returns per asset | From forecasting tools (ARIMA, LSTM) or historical averages |
| `Sigma` | matrix (N, N) | Covariance matrix of asset returns | From historical returns, GARCH, or factor models |
| `risk_aversion` or `target_return` | float | Risk aversion parameter or target return | Risk policy or investor preference |
| `constraints` | dict | Portfolio constraints (long-only, sector limits, etc.) | Investment policy |
| `risk_free_rate` | float | Risk-free rate for Sharpe ratio | From macro data (FRED, RBI) |

**Constraint Types:**
- Long-only: weights >= 0
- Budget: sum(weights) = 1
- Sector limits: sum(weights[sector]) <= limit
- Turnover: ||w_new - w_old|| <= turnover_limit

---

### 1.3 Computation / Algorithm Steps

1. **Estimate Inputs**
   - Compute expected returns `mu` from forecasting models or historical averages
   - Estimate covariance matrix `Sigma` from historical returns (with optional shrinkage)
   - Set risk aversion parameter or target return

2. **Formulate Optimization Problem**
   - **Maximize Sharpe Ratio:**
     $$
     \max_w \frac{w^\top \mu - r_f}{\sqrt{w^\top \Sigma w}}
     $$
   - **Minimize Variance (for given return):**
     $$
     \min_w w^\top \Sigma w \quad \text{s.t.} \quad w^\top \mu = \mu_{\text{target}}, \quad \sum w_i = 1
     $$
   - **Maximize Utility:**
     $$
     \max_w w^\top \mu - \lambda w^\top \Sigma w \quad \text{s.t.} \quad \sum w_i = 1, \quad w_i \geq 0
     $$

3. **Solve Quadratic Program**
   - Use `cvxpy`, `scipy.optimize`, or `quadprog`
   - Handle constraints (equality, inequality, bounds)
   - Return optimal weights

4. **Post-Processing**
   - Validate constraints are satisfied
   - Compute portfolio statistics (expected return, volatility, Sharpe ratio)
   - Generate efficient frontier (optional)

5. **Integration into Pipeline**
   - Run daily/weekly with updated `mu` and `Sigma`
   - Feed optimal weights to execution layer
   - Log all inputs, outputs, constraints to MLflow

---

### 1.4 Usage in Hedge Fund Context

* **Baseline optimizer**: Compare Markowitz against Black-Litterman, CVaR optimization, quantum methods (QMV, QAOA)
* **Efficient frontier analysis**: Generate and visualize risk-return tradeoffs
* **Constraint testing**: Test impact of different constraints (sector limits, turnover) on optimal portfolios
* **Regime-adaptive**: Use regime-specific `mu` and `Sigma` from regime detection modules

---

## 2. Why This Tool is a Sound Choice (with Recent Literature)

Markowitz Mean-Variance remains the cornerstone of portfolio optimization:

* **Markowitz (1952)** - "Portfolio Selection" establishes the mean-variance framework, foundational to modern portfolio theory. ([Journal of Finance][1])
* **Michaud (1989)** - "The Markowitz Optimization Enigma: Is 'Optimized' Optimal?" discusses estimation error challenges. ([Financial Analysts Journal][2])
* **DeMiguel et al. (2009)** - "Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?" shows estimation challenges but validates framework. ([Review of Financial Studies][3])
* **Ledoit & Wolf (2004)** - "Honey, I Shrunk the Sample Covariance Matrix" addresses estimation error via shrinkage. ([Journal of Portfolio Management][4])
* **Recent extensions**: Robust optimization, factor models, and quantum variants (QMV) extend the framework.

**Caveats:**
* **Estimation error** in `mu` and `Sigma` can severely degrade performance
* Assumes **quadratic utility** (may not capture tail risk preferences)
* **Sensitivity to inputs**: small changes in `mu` can lead to large weight changes
* **No consideration of tail risk**: CVaR optimization addresses this

In summary: Markowitz is the **essential baseline** that provides interpretable, theoretically sound portfolio construction and serves as benchmark for advanced methods.

---

## 3. Example Pseudocode (Pythonic)

```python
import numpy as np
import cvxpy as cp

def markowitz_optimize(mu: np.ndarray, Sigma: np.ndarray,
                      risk_aversion: float = 1.0,
                      risk_free_rate: float = 0.0,
                      constraints: dict = None):
    """
    Solve Markowitz mean-variance optimization.
    
    mu: (N,) expected returns
    Sigma: (N, N) covariance matrix
    risk_aversion: lambda parameter (higher = more risk averse)
    risk_free_rate: risk-free rate for Sharpe ratio
    constraints: dict with 'long_only', 'budget', 'sector_limits', etc.
    """
    N = len(mu)
    w = cp.Variable(N)
    
    # Objective: maximize utility = mu^T w - lambda * w^T Sigma w
    objective = cp.Maximize(mu @ w - risk_aversion * cp.quad_form(w, Sigma))
    
    # Constraints
    constraint_list = [cp.sum(w) == 1]  # Budget constraint
    
    if constraints is None:
        constraints = {}
    
    if constraints.get('long_only', True):
        constraint_list.append(w >= 0)
    
    if 'sector_limits' in constraints:
        for sector, limit in constraints['sector_limits'].items():
            constraint_list.append(cp.sum(w[sector]) <= limit)
    
    if 'turnover_limit' in constraints:
        w_old = constraints.get('w_old', np.zeros(N))
        constraint_list.append(cp.norm(w - w_old, 1) <= constraints['turnover_limit'])
    
    # Solve
    problem = cp.Problem(objective, constraint_list)
    problem.solve()
    
    if problem.status != 'optimal':
        raise ValueError(f"Optimization failed: {problem.status}")
    
    w_opt = w.value
    
    # Compute portfolio statistics
    portfolio_return = mu @ w_opt
    portfolio_vol = np.sqrt(w_opt @ Sigma @ w_opt)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
    
    return {
        'weights': w_opt,
        'expected_return': portfolio_return,
        'volatility': portfolio_vol,
        'sharpe_ratio': sharpe_ratio,
        'status': problem.status
    }
```

---

## 4. How You Can Improve / Extend (2–3 research directions)

1. **Robust Markowitz**: Use robust optimization techniques (worst-case `mu`, `Sigma`) or shrinkage estimators (Ledoit-Wolf) to handle estimation error.
2. **Factor-Based Markowitz**: Replace full covariance `Sigma` with factor model (Fama-French, PCA) to reduce dimensionality and improve stability.
3. **Quantum Mean-Variance (QMV) Comparison**: Compare classical Markowitz against quantum optimization (QMV, QAOA), exploring when quantum methods provide advantage for large-scale problems.

[1]: https://www.jstor.org/stable/2327556 "Portfolio Selection"
[2]: https://www.jstor.org/stable/4479057 "The Markowitz Optimization Enigma: Is 'Optimized' Optimal?"
[3]: https://academic.oup.com/rfs/article/22/5/1915/1593121 "Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?"
[4]: https://www.jstor.org/stable/4485060 "Honey, I Shrunk the Sample Covariance Matrix"

