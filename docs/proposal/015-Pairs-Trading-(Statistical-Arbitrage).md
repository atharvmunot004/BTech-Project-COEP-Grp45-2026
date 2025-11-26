## 1. Methodology for Implementation

### 1.1 Overview

Pairs Trading (Statistical Arbitrage) is a market-neutral trading strategy that exploits temporary mispricings between two correlated assets. The strategy involves identifying pairs of assets with historically stable relationships, monitoring their price spread, and trading when the spread deviates from its historical mean. In your hedge fund system, this will serve as a **classical trading strategy** that can be enhanced with ML/quantum methods for pair selection and signal generation.

Key components:
- **Pair selection**: Identify cointegrated or highly correlated asset pairs
- **Spread calculation**: Compute normalized spread (z-score) between pair prices
- **Entry/exit signals**: Trade when spread deviates beyond thresholds
- **Risk management**: Position sizing, stop-loss, and portfolio-level risk controls

---

### 1.2 Inputs & How to Obtain Them

| Input | Type / Shape | Description | Source / Estimation |
|-------|-------------|-------------|-------------------|
| `price_series_1`, `price_series_2` | array (T,) | Price series for two assets | From market data ingestion |
| `lookback_window` | integer | Historical window for spread calculation | Hyperparameter (e.g., 60, 120, 252 days) |
| `entry_threshold` | float | Z-score threshold for entry signal | Hyperparameter (e.g., 2.0, 2.5 standard deviations) |
| `exit_threshold` | float | Z-score threshold for exit signal | Hyperparameter (e.g., 0.5, 1.0 standard deviations) |
| `hedge_ratio` | float | Optimal hedge ratio (beta) | Estimated via OLS regression or Kalman filter |
| `cointegration_test` | boolean | Whether to test for cointegration | Statistical test (Engle-Granger, Johansen) |

---

### 1.3 Computation / Algorithm Steps

1. **Pair Selection**
   - Identify candidate pairs (same sector, correlated assets, etc.)
   - Test for cointegration (Engle-Granger test) or high correlation
   - Select pairs with stable historical relationship

2. **Hedge Ratio Estimation**
   - Estimate optimal hedge ratio via OLS: `spread = price_1 - beta * price_2`
   - Or use dynamic hedge ratio via Kalman filter
   - Compute spread: `spread_t = price_1_t - beta * price_2_t`

3. **Spread Normalization**
   - Compute rolling mean and standard deviation of spread
   - Calculate z-score: `z_t = (spread_t - mean_t) / std_t`

4. **Signal Generation**
   - **Entry (long spread)**: When z-score < -entry_threshold (spread too low, expect reversion)
   - **Entry (short spread)**: When z-score > entry_threshold (spread too high, expect reversion)
   - **Exit**: When z-score crosses exit_threshold (spread reverts)

5. **Position Sizing & Risk Management**
   - Size positions based on volatility and risk budget
   - Set stop-loss levels
   - Monitor portfolio-level exposure

6. **Backtesting & Validation**
   - Walk-forward backtest
   - Compute performance metrics (Sharpe ratio, max drawdown, win rate)
   - Compare against buy-and-hold and other strategies

7. **Integration into Pipeline**
   - Run daily to identify trading opportunities
   - Generate signals and feed to execution layer
   - Log all pairs, signals, positions, performance to MLflow

---

### 1.4 Usage in Hedge Fund Context

* **Market-neutral strategy**: Provides diversification and reduces market exposure
* **ML-enhanced pair selection**: Use ML/quantum methods to identify optimal pairs
* **Regime-aware**: Adapt thresholds or pause trading during regime shifts
* **Portfolio of pairs**: Trade multiple pairs simultaneously for diversification
* **Research**: Compare classical pairs trading vs ML/quantum-enhanced variants

---

## 2. Why This Tool is a Sound Choice (with Recent Literature)

Pairs trading is a well-established quantitative strategy:

* **Gatev et al. (2006)** - "Pairs Trading: Performance of a Relative-Value Arbitrage Rule" provides foundational analysis. ([Review of Financial Studies][1])
* **Elliott et al. (2005)** - "Pairs Trading" establishes theoretical framework. ([Quantitative Finance][2])
* **Vidyamurthy (2004)** - "Pairs Trading: Quantitative Methods and Analysis" offers practical guide. ([Wiley][3])
* **Do & Faff (2010)** - "Does simple pairs trading still work?" examines strategy performance. ([Financial Analysts Journal][4])
* **Recent extensions**: ML-based pair selection, regime-aware trading, and quantum optimization for portfolio of pairs.

**Advantages:**
* Market-neutral (reduces market risk)
* Exploits mean reversion
* Can be profitable in sideways markets

**Caveats:**
* Requires cointegration/correlation to persist
* Transaction costs can erode profits
* Risk of permanent divergence (pair breaks down)
* Requires careful risk management

---

## 3. Example Pseudocode (Pythonic)

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS

def pairs_trading_signals(price_1: np.ndarray, price_2: np.ndarray,
                         lookback_window: int = 120,
                         entry_threshold: float = 2.0,
                         exit_threshold: float = 0.5):
    """
    Generate pairs trading signals.
    
    price_1, price_2: (T,) price series for two assets
    lookback_window: historical window for spread calculation
    entry_threshold: z-score threshold for entry
    exit_threshold: z-score threshold for exit
    """
    T = len(price_1)
    
    # Test for cointegration
    score, pvalue, _ = coint(price_1, price_2)
    is_cointegrated = pvalue < 0.05
    
    # Estimate hedge ratio
    model = OLS(price_1, price_2).fit()
    hedge_ratio = model.params[0]
    
    # Compute spread
    spread = price_1 - hedge_ratio * price_2
    
    # Compute rolling z-score
    signals = []
    positions = []
    
    for t in range(lookback_window, T):
        spread_window = spread[t-lookback_window:t]
        spread_mean = np.mean(spread_window)
        spread_std = np.std(spread_window)
        
        z_score = (spread[t] - spread_mean) / spread_std if spread_std > 0 else 0
        
        # Generate signals
        if z_score < -entry_threshold:
            signal = 'long_spread'  # Buy asset 1, short asset 2
        elif z_score > entry_threshold:
            signal = 'short_spread'  # Short asset 1, buy asset 2
        elif abs(z_score) < exit_threshold:
            signal = 'exit'
        else:
            signal = 'hold'
        
        signals.append(signal)
        positions.append({
            'z_score': z_score,
            'spread': spread[t],
            'hedge_ratio': hedge_ratio
        })
    
    return {
        'signals': signals,
        'positions': positions,
        'is_cointegrated': is_cointegrated,
        'hedge_ratio': hedge_ratio,
        'spread': spread
    }
```

---

## 4. How You Can Improve / Extend (2–3 research directions)

1. **ML-Enhanced Pair Selection**: Use machine learning (LSTM, tree methods) or quantum methods to identify optimal pairs and predict spread movements, going beyond simple cointegration tests.
2. **Regime-Aware Pairs Trading**: Use regime detection (HMM, GMM) to adapt entry/exit thresholds or pause trading during regime shifts when relationships may break down.
3. **Portfolio of Pairs Optimization**: Use portfolio optimization (Markowitz, CVaR, quantum methods) to construct optimal portfolio of multiple pairs, balancing risk and return.

[1]: https://academic.oup.com/rfs/article/19/3/797/1599809 "Pairs Trading: Performance of a Relative-Value Arbitrage Rule"
[2]: https://www.tandfonline.com/doi/abs/10.1080/14697680500149370 "Pairs Trading"
[3]: https://www.wiley.com/en-us/Pairs+Trading%3A+Quantitative+Methods+and+Analysis-p-9780471460671 "Pairs Trading: Quantitative Methods and Analysis"
[4]: https://www.tandfonline.com/doi/abs/10.2469/faj.v66.n4.3 "Does simple pairs trading still work?"

