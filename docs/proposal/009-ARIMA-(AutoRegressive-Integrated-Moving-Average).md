## 1. Methodology for Implementation

### 1.1 Overview

ARIMA (AutoRegressive Integrated Moving Average) is a classical time-series forecasting model that combines autoregressive (AR), differencing (I), and moving average (MA) components. In your hedge fund system, ARIMA will serve as a **baseline forecasting tool** for predicting asset returns, volatility, or other financial time series. It provides interpretable, parametric forecasts that can be compared against more complex models (LSTM, quantum methods) in your comparative study.

ARIMA models are particularly useful for:
- Short-to-medium-term return forecasting
- Volatility prediction (when applied to squared returns)
- Baseline comparison against machine learning and quantum methods
- Regime-adaptive forecasting when combined with regime detection

---

### 1.2 Inputs & How to Obtain Them

| Input | Type / Shape | Description | Source / Estimation |
|-------|-------------|-------------|-------------------|
| `returns` or `price_series` | vector (T,) | Time series of returns or prices | From market data ingestion (yfinance/Polygon) |
| `p` | integer | Autoregressive order (AR) | Selected via AIC/BIC or auto_arima |
| `d` | integer | Differencing order (I) | Selected via ADF test or auto_arima |
| `q` | integer | Moving average order (MA) | Selected via AIC/BIC or auto_arima |
| `seasonal` | boolean or dict | Seasonal ARIMA (SARIMA) parameters | Optional, for intraday/weekly patterns |
| `forecast_horizon` | integer | Number of steps ahead to forecast | Risk/strategy policy (e.g., 1, 5, 10 days) |
| `training_window` | integer | Historical window for model fitting | Policy (e.g., 250, 500 trading days) |

**Model Selection:**
- Use **auto_arima** (pmdarima) to automatically select (p, d, q) orders
- Or manually test via AIC/BIC information criteria
- Perform **Augmented Dickey-Fuller (ADF) test** to determine differencing order `d`

---

### 1.3 Computation / Algorithm Steps

1. **Data Preprocessing**
   - Fetch historical price/return series
   - Check for stationarity (ADF test)
   - Apply differencing if needed (determine `d`)
   - Handle missing values and outliers

2. **Model Selection**
   - Grid search or auto_arima to select optimal (p, d, q)
   - Minimize AIC or BIC
   - Validate on hold-out set

3. **Model Fitting**
   - Fit ARIMA(p, d, q) model to training data
   - Estimate parameters via maximum likelihood
   - Check residuals for white noise (Ljung-Box test)

4. **Forecasting**
   - Generate point forecasts for `forecast_horizon` steps ahead
   - Compute prediction intervals (confidence bands)
   - Return forecast mean and variance

5. **Validation & Backtesting**
   - Walk-forward validation
   - Compute forecast errors (MAE, RMSE, MAPE)
   - Compare against naive baseline (random walk)

6. **Integration into Pipeline**
   - Run daily/weekly to update forecasts
   - Feed forecasts into portfolio optimizer (expected returns)
   - Log all parameters, forecasts, errors to MLflow

---

### 1.4 Usage in Hedge Fund Context

* Use **ARIMA forecasts as expected returns** input to Markowitz, Black-Litterman, or other optimizers
* Use **ARIMA for volatility forecasting** (ARIMA-GARCH hybrid)
* **Baseline comparison**: compare ARIMA forecast accuracy vs LSTM, quantum methods under different regimes
* **Regime-adaptive**: fit separate ARIMA models per regime (detected by HMM/GMM)

---

## 2. Why This Tool is a Sound Choice (with Recent Literature)

ARIMA remains a cornerstone of time-series forecasting in finance due to its interpretability, statistical foundation, and proven track record:

* **Hyndman & Athanasopoulos (2021)** - "Forecasting: principles and practice" establishes ARIMA as fundamental baseline for time-series forecasting. ([OTexts][1])
* **Box & Jenkins (1976)** - Original ARIMA methodology remains standard in econometrics and finance. ([Wiley][2])
* **Pai & Lin (2005)** - "Using ARIMA model to forecast stock price" demonstrates ARIMA's applicability to financial time series. ([ScienceDirect][3])
* **Contreras et al. (2003)** - "ARIMA models to predict next-day electricity prices" shows ARIMA's utility in financial-like forecasting problems. ([IEEE][4])
* **Recent extensions**: ARIMA-GARCH hybrids, regime-switching ARIMA, and auto_arima tools make ARIMA practical for modern applications.

**Caveats:**
* ARIMA assumes **linear relationships** and may miss nonlinear patterns captured by ML models
* **Parameter stability** can degrade in non-stationary or regime-shifting markets
* **Long-horizon forecasts** tend to revert to mean, limiting utility for distant predictions

In summary: ARIMA is a **valuable, interpretable baseline** that provides statistical rigor and serves as a benchmark against which more complex models are evaluated.

---

## 3. Example Pseudocode (Pythonic)

```python
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

def arima_forecast(returns: np.ndarray, forecast_horizon: int = 1, 
                   training_window: int = 250):
    """
    Fit ARIMA model and generate forecasts.
    
    returns: (T,) time series of returns
    forecast_horizon: steps ahead to forecast
    training_window: historical window for training
    """
    # Use auto_arima to select optimal (p, d, q)
    model = auto_arima(returns[-training_window:], 
                       seasonal=False,
                       stepwise=True,
                       suppress_warnings=True,
                       error_action='ignore')
    
    # Get selected orders
    p, d, q = model.order
    
    # Fit model
    fitted_model = ARIMA(returns[-training_window:], order=(p, d, q))
    fitted_model = fitted_model.fit()
    
    # Generate forecasts
    forecast = fitted_model.forecast(steps=forecast_horizon)
    forecast_ci = fitted_model.get_forecast(steps=forecast_horizon).conf_int()
    
    # Validate residuals
    residuals = fitted_model.resid
    ljung_box = acorr_ljungbox(residuals, lags=10, return_df=True)
    
    return {
        'forecast': forecast,
        'forecast_ci': forecast_ci,
        'model_order': (p, d, q),
        'aic': fitted_model.aic,
        'bic': fitted_model.bic,
        'residuals': residuals,
        'ljung_box_pvalue': ljung_box['lb_pvalue'].iloc[-1]
    }
```

---

## 4. How You Can Improve / Extend (2–3 research directions)

1. **Regime-Switching ARIMA**: Use your regime detection (HMM, GMM) to fit separate ARIMA models per regime, switching models based on current regime state.
2. **ARIMA-GARCH Hybrid**: Combine ARIMA for mean forecasting with GARCH for volatility modeling, capturing both return dynamics and volatility clustering.
3. **Ensemble ARIMA-Quantum**: Use ARIMA as classical baseline and combine with quantum forecasting methods (Q-LSTM, QNN) in ensemble, comparing performance under different market conditions.

[1]: https://otexts.com/fpp3/arima.html "ARIMA models - Forecasting: Principles and Practice"
[2]: https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+5th+Edition-p-9781118675021 "Time Series Analysis: Forecasting and Control"
[3]: https://www.sciencedirect.com/science/article/pii/S0305054805000081 "Using ARIMA model to forecast stock price"
[4]: https://ieeexplore.ieee.org/document/1238291 "ARIMA models to predict next-day electricity prices"

