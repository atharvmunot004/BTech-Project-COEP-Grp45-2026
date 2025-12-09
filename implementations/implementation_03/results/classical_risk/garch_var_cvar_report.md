# GARCH(1,1) Volatility Forecasting for VaR/CVaR Evaluation Report

**Generated:** 2025-12-10 00:27:46

## Methodology Overview

### GARCH(1,1) Volatility Forecasting for VaR and CVaR

GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models capture
time-varying volatility, which is essential for accurate risk assessment.
VaR and CVaR are derived from GARCH volatility forecasts using distributional assumptions.

**Advantages:**
- Captures volatility clustering (high volatility followed by high volatility)
- Adapts to changing market conditions
- Provides dynamic risk estimates
- Well-established in financial risk management

## GARCH Model Description

### GARCH(1,1) Model

The GARCH(1,1) model specifies the conditional variance as:

σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁

Where:
- σ²ₜ is the conditional variance at time t
- ω is the constant term
- α captures the ARCH effect (lagged squared residuals)
- β captures the GARCH effect (lagged conditional variance)
- εₜ are the residuals

### Model Parameters

- **GARCH Order (p):** 1
- **ARCH Order (q):** 1
- **Distribution:** normal
- **Mean Model:** Zero
- **Return Type:** log

## Volatility Forecast Method

### Rolling Window GARCH Forecasting

For each date in the evaluation period:
1. Fit GARCH(1,1) model to a rolling window of historical returns
2. Forecast conditional volatility for the next horizon periods
3. Use forecasted volatility to compute VaR and CVaR

### Forecast Settings

- **Forecast Method:** rolling
- **Estimation Windows:** [252, 500] days
- **Forecast Horizons:** [1, 10] days
- **Fallback to Long-Run Variance:** True

## VaR/CVaR Derivation from GARCH

### From Volatility to Risk Measures

**VaR Calculation:**
- VaR = σₜ₊ₕ · z_α · √h
- Where σₜ₊ₕ is the forecasted volatility, z_α is the quantile from the assumed distribution, and h is the horizon

**CVaR Calculation:**
- CVaR = σₜ₊ₕ · ES_α · √h
- Where ES_α is the Expected Shortfall factor for the assumed distribution

**Distribution Assumptions:**
- Normal distribution: VaR and CVaR computed using standard normal quantiles

- **Confidence Levels:** [0.95, 0.99]

## Summary Statistics

- **Total Portfolio-Configuration Combinations:** 800

## Backtesting Results

- **Average Hit Rate:** 0.0000

- **Average Violation Ratio:** 0.0000
  - Ratio > 1 indicates overestimation of risk
  - Ratio < 1 indicates underestimation of risk

### Traffic Light Zones

- **Green:** 800 configurations (100.0%)

- **Kupiec Test Passed:** 0/800 configurations (0.0%)

## Tail Risk Analysis

- **Average Mean Exceedance (VaR):** nan

- **Average Max Exceedance (VaR):** nan

- **Average CVaR Mean Exceedance:** nan

- **Average CVaR Max Exceedance:** nan

## Portfolio Structure Effects

- **Average Number of Active Assets:** 6.04

- **Average HHI Concentration:** 0.3363

## Robustness and Normality Checks

- **Average Skewness:** -0.9134

- **Average Excess Kurtosis:** 15.4149

- **Normality Tests Passed (Jarque-Bera):** 0/800 configurations (0.0%)

## Computational Performance

- **Average GARCH Fitting Time per Configuration:** 149117.06 ms

- **Average Forecast Time per Configuration:** 149117.06 ms

- **Average Total Runtime per Portfolio:** 1192936.49 ms

## Key Insights

### Findings

- **Risk Underestimation:** GARCH-based VaR tends to underestimate risk

### Recommendations

- GARCH models are well-suited for portfolios with volatility clustering
- Consider t-distribution for portfolios with fat-tailed returns
- Monitor portfolios in 'red' traffic light zone more closely
- Adjust estimation windows based on market regime

## Detailed Metrics

### Summary Statistics by Metric

```
       portfolio_id  confidence_level     horizon  estimation_window  var_runtime_ms  garch_fitting_time_ms  forecast_time_ms  hit_rate  violation_ratio  kupiec_unconditional_coverage  kupiec_test_statistic  christoffersen_independence  christoffersen_independence_statistic  christoffersen_conditional_coverage  christoffersen_conditional_coverage_statistic  num_violations  total_observations  expected_violations  mean_exceedance  max_exceedance  std_exceedance  quantile_loss_score  rmse_var_vs_losses  cvar_mean_exceedance  cvar_max_exceedance  cvar_std_exceedance  portfolio_size  num_active_assets  hhi_concentration  effective_number_of_assets  covariance_condition_number    skewness    kurtosis  jarque_bera_p_value  jarque_bera_statistic  runtime_per_portfolio_ms  p95_runtime_ms  mean_runtime_ms  median_runtime_ms  min_runtime_ms  max_runtime_ms
count    800.000000        800.000000  800.000000         800.000000      800.000000             800.000000        800.000000     800.0            800.0                            0.0                    0.0                          0.0                                    0.0                                  0.0                                            0.0           800.0          800.000000           800.000000              0.0             0.0             0.0                  0.0                 0.0                   0.0                  0.0                  0.0      800.000000         800.000000         800.000000                  800.000000                 8.000000e+02  800.000000  800.000000                800.0             800.000000              8.000000e+02    8.000000e+02     8.000000e+02       8.000000e+02      800.000000    8.000000e+02
mean      49.500000          0.970000    5.500000         376.000000   149117.061323          149117.061323     149117.061323       0.0              0.0                            NaN                    NaN                          NaN                                    NaN                                  NaN                                            NaN             0.0         2095.000000            62.850000              NaN             NaN             NaN                  NaN                 NaN                   NaN                  NaN                  NaN        6.040000           6.040000           0.336270                    3.696964                 1.188948e+01   -0.913371   15.414902                  0.0           22508.902192              1.192936e+06    1.330041e+06     1.192936e+06       1.195731e+06   554440.096617    1.390645e+06
std       28.884128          0.020013    4.502815         124.077573    19737.973576           19737.973576      19737.973576       0.0              0.0                            NaN                    NaN                          NaN                                    NaN                                  NaN                                            NaN             0.0          124.077573            42.164216              NaN             NaN             NaN                  NaN                 NaN                   NaN                  NaN                  NaN        2.538988           2.538988           0.178299                    1.559267                 1.777468e-15    0.469266    3.975490                  0.0           11021.759460              0.000000e+00    0.000000e+00     0.000000e+00       0.000000e+00        0.000000    0.000000e+00
min        0.000000          0.950000    1.000000         252.000000    57921.334982           57921.334982      57921.334982       0.0              0.0                            NaN                    NaN                          NaN                                    NaN                                  NaN                                            NaN             0.0         1971.000000            19.710000              NaN             NaN             NaN                  NaN                 NaN                   NaN                  NaN                  NaN        2.000000           2.000000           0.142291                    1.065007                 1.188948e+01   -1.631473    4.699076                  0.0            1838.566862              1.192936e+06    1.330041e+06     1.192936e+06       1.195731e+06   554440.096617    1.390645e+06
25%       24.750000          0.950000    1.000000         252.000000   144593.844891          144593.844891     144593.844891       0.0              0.0                            NaN                    NaN                          NaN                                    NaN                                  NaN                                            NaN             0.0         1971.000000            21.570000              NaN             NaN             NaN                  NaN                 NaN                   NaN                  NaN                  NaN        4.000000           4.000000           0.209405                    2.367240                 1.188948e+01   -1.264798   12.296512                  0.0           13017.186352              1.192936e+06    1.330041e+06     1.192936e+06       1.195731e+06   554440.096617    1.390645e+06
50%       49.500000          0.970000    5.500000         376.000000   150584.631920          150584.631920     150584.631920       0.0              0.0                            NaN                    NaN                          NaN                                    NaN                                  NaN                                            NaN             0.0         2095.000000            60.370000              NaN             NaN             NaN                  NaN                 NaN                   NaN                  NaN                  NaN        6.000000           6.000000           0.279022                    3.583950                 1.188948e+01   -0.968824   15.794671                  0.0           22148.386120              1.192936e+06    1.330041e+06     1.192936e+06       1.195731e+06   554440.096617    1.390645e+06
75%       74.250000          0.990000   10.000000         500.000000   157203.700483          157203.700483     157203.700483       0.0              0.0                            NaN                    NaN                          NaN                                    NaN                                  NaN                                            NaN             0.0         2219.000000           101.650000              NaN             NaN             NaN                  NaN                 NaN                   NaN                  NaN                  NaN        8.000000           8.000000           0.422513                    4.776810                 1.188948e+01   -0.679904   18.061510                  0.0           29329.364478              1.192936e+06    1.330041e+06     1.192936e+06       1.195731e+06   554440.096617    1.390645e+06
max       99.000000          0.990000   10.000000         500.000000   209586.912394          209586.912394     209586.912394       0.0              0.0                            NaN                    NaN                          NaN                                    NaN                                  NaN                                            NaN             0.0         2219.000000           110.950000              NaN             NaN             NaN                  NaN                 NaN                   NaN                  NaN                  NaN       10.000000          10.000000           0.938961                    7.027832                 1.188948e+01    0.667369   23.596238                  0.0           52415.819664              1.192936e+06    1.330041e+06     1.192936e+06       1.195731e+06   554440.096617    1.390645e+06
```
