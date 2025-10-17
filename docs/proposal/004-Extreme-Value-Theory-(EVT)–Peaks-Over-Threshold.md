## 1. Methodology Overview

EVT (Extreme Value Theory) focuses only on the **tail behavior** of loss (or return) distributions. The POT (Peaks Over Threshold) method models the **excesses above a high threshold** using a **Generalized Pareto Distribution (GPD)**. In finance, this is valuable because we care most about extreme losses, not the entire distribution.

In your system, the EVT/POT tool will:
- Estimate tail-based VaR / ES robustly in high-confidence regimes (e.g. 99.9%)  
- Act as a “tail-adjustment” complement to parametric and Monte Carlo tools  
- Provide regime-conditional extreme risk estimates  
- Serve as a benchmark in your comparative study especially for crises periods  

---

## 2. Inputs & Variables (and How to Get Them)

| Input | Type / Shape | Description | How to Obtain / Estimate |
|---|---|-----------------------|---------------------------|
| `L_t` | vector, length T | Loss series (or negative returns) for the portfolio | Compute from historical returns and weights: \( L_t = -w^\top r_t \) |
| `u` | scalar threshold | High threshold beyond which you treat data as “extreme” | Choose via percentile (e.g. 95th, 98th) or via automated threshold selection methods |
| `excesses` | vector | Observations \( y_i = L_t - u \) for all \( L_t > u \) | Filter loss data > u |
| `k` | integer | Number of exceedances (size of excesses) | Count of data points > u |
| GPD parameters: `ξ` (xi) and `β` (beta) | scalars | Tail shape and scale parameters of the GPD model | Estimate via Maximum Likelihood (or alternative robust methods) on excesses |
| Confidence level `α` | float (e.g. 0.95, 0.99, 0.995) | The tail quantile to compute VaR / ES | Risk policy setting |
| Horizon scaling `T_horizon` | float or integer | Time scaling (days, years) | Based on your risk horizon (1-day, 5-day) |
| (Optional) Covariates / regression inputs | vector or matrix | Explanatory variables (e.g. volatility regime, market factors) for tail parameters | From your feature pipeline / regime detector |

---

## 3. Implementation Steps

### 3.1 Preprocessing & Loss Series

1. Compute portfolio returns \( r_t \) and transform to losses \( L_t = - w^\top r_t \).  
2. Optionally, standardize losses by current volatility (e.g. GARCH-normalized residuals) to reduce heteroscedasticity.

### 3.2 Threshold Selection

Choosing `u` is crucial:

- Traditional approach: set `u` as high percentile (95th, 98th) of the loss distribution.  
- Diagnostic tools: mean excess plot (plot average excesses vs threshold), threshold stability plots (parameter estimates vs `u`).  
- More advanced: automatic threshold selection (e.g. minimizing bias/variance tradeoff) or methods proposed in *“Efficient Estimation in Extreme Value Regression Models”*. citeturn0search0turn0search7  
- Recent work like Benito et al. (2023) emphasizes assessing how threshold choice affects tail risk estimates. citeturn0search5  

### 3.3 Fit the GPD to Excesses

Given exceedances \( y_i = L_i - u \):

- The GPD’s CDF:  
  $$
  G(y) = 1 - \left(1 + \frac{\xi y}{\beta}\right)^{-1/\xi}, \quad \text{for } \xi \neq 0  
  $$
- Perform Maximum Likelihood Estimation to find `ξ` and `β`.  
- If using covariates (tail regression), you can let `ξ` and/or `β` be functions of features. Hambuckers et al. (2023) propose joint estimation of threshold and GPD parameters. citeturn0search7  
- Use robust estimation or censored likelihood if sample is small or tail is noisy.

### 3.4 Compute Tail VaR and ES

Once parameters are estimated:

- **VaR (at level α):**  
  $$
  \text{VaR}_\alpha = u + \frac{\beta}{\xi} \left( \left(\frac{ (k / n) }{1 - \alpha} \right)^{\xi} - 1 \right)
  $$  
  where \( n \) is total data size, \( k \) is number of exceedances.

- **ES / Expected Shortfall:**  
  $$
  \text{ES}_\alpha = \frac{ \text{VaR}_\alpha + \beta - \xi u }{1 - \xi } \quad \text{(for }\xi < 1\text{)}
  $$  
  Another formula:  
  $$
  \text{ES}_\alpha = \frac{ \text{VaR}_\alpha }{1 - \xi} + \frac{ \beta - \xi u }{1 - \xi }
  $$

- Scale for multi-day horizon if needed (carefully — tail scaling may not follow \(\sqrt{T}\)).

### 3.5 Diagnostics & Backtesting

- Check goodness-of-fit: QQ-plots of exceedances, probability plots.  
- Stability of parameter estimates across thresholds.  
- Backtest via number of violations beyond VaR, ES exceedances.  
- Use **Kupiec** and **Christoffersen tests** on tail estimates.  
- Compare estimated VaR/ES against classical and Monte Carlo versions in different regimes.

---

## 4. Pseudocode Example (Python-like)

```python
import numpy as np
from scipy.stats import genpareto
from scipy.optimize import minimize

def fit_gpd_excesses(excesses):
    # excesses = y_i = losses - u
    # Initialize xi, beta
    def neg_loglik(params):
        xi, beta = params
        if beta <= 0:
            return np.inf
        return -np.sum(genpareto.logpdf(excesses, c=xi, scale=beta))
    res = minimize(neg_loglik, x0=[0.1, np.std(excesses)], bounds=[(-1,2), (1e-6, None)])
    xi, beta = res.x
    return xi, beta

def evt_var_es(losses, u, alpha):
    # losses: array of loss values
    exceed = losses[losses > u]
    excesses = exceed - u
    xi, beta = fit_gpd_excesses(excesses)
    n = len(losses)
    k = len(exceed)
    # VaR
    frac = k / n
    var = u + (beta / xi) * ( (frac / (1 - alpha))**xi - 1 )
    # ES
    es = var / (1 - xi) + (beta - xi * u) / (1 - xi)
    return var, es, xi, beta
```

---

## 5. Why EVT / POT is a Sound Choice for Hedge Funds (with Recent Literature)

- EVT concentrates modeling power where it matters — in the tail — rather than diluting it over the whole distribution.  
- In stress/crisis periods, tail behavior deviates dramatically from normal/parametric assumptions; EVT is more robust.  
- Realized-POT methods (using high-frequency data) improve tail forecasts beyond daily-only models. Bee et al. propose “Realized POT” using intraday data. citeturn0search4  
- An empirical review (Candia et al., 2024) surveys dynamic threshold exceedance models and highlights that EVT-based models often outperform conventional ones in extreme risk forecasting. citeturn0search2  
- The threshold choice problem is recognized as critical; Benito et al. (2023) analyze how threshold selection affects risk estimates. citeturn0search5  
- Hambuckers et al. (2023) propose a new method for simultaneous threshold and tail parameter estimation, reducing subjectivity in threshold choice. citeturn0search7  
- Tomlinson et al. (2022) compare conditional EVT models (2T-POT Hawkes) and show improved forecasting for left/right tail quantiles, demonstrating better extreme quantile forecasts than GARCH-EVT hybrids. citeturn0academia10  

Thus, for hedge funds with tail sensitivity (large positions, leverage, drawdown risk), EVT/POT is a critical complementary tool.

---

## 6. Extensions / Research Directions

1. **Regime-adaptive POT**: let threshold \( u \) and tail parameters vary by regime (via HMM / QBM).  
2. **Dynamic threshold estimation**: infer \( u \) jointly with GPD parameters (as in Hambuckers et al.). citeturn0search7  
3. **Integrate high-frequency data (Realized POT / RPOT)**: use intraday volatility measures to condition tail models. citeturn0search4  
