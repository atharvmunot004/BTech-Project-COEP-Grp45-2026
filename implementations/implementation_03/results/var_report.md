# Variance-Covariance Value-at-Risk Evaluation Report

**Generated:** 2025-12-09 16:38:04

## Methodology Overview

### Variance-Covariance (Parametric) VaR

The Variance-Covariance method assumes that returns follow a normal distribution.
VaR is calculated as:

```
VaR = -μ - z_α * σ * √(horizon)
```

Where:
- μ = mean return
- σ = standard deviation of returns
- z_α = z-score for confidence level α
- horizon = time horizon in days

### VaR Settings

- **Confidence Levels:** [0.95, 0.99]
- **Horizons:** [1, 10] days
- **Estimation Windows:** [252] days

## Summary Statistics

- **Total Portfolios Evaluated:** 400

## Backtesting Results

- **Average Hit Rate:** 0.8864

- **Average Violation Ratio:** 53.4297
  - Ratio > 1 indicates overestimation of risk
  - Ratio < 1 indicates underestimation of risk

### Traffic Light Zones

- **Red:** 400 portfolios (100.0%)

- **Kupiec Test Passed:** 0/400 portfolios (0.0%)

## Tail Risk Analysis

- **Average Mean Exceedance:** 0.048713

- **Average Max Exceedance:** 0.210774

## Portfolio Structure Effects

- **Average Number of Active Assets:** 6.04

- **Average HHI Concentration:** 0.3363
  - Higher values indicate more concentrated portfolios

- **Average Effective Number of Assets:** 3.70

## Robustness and Normality Checks

- **Average Skewness:** -0.8043
  - Negative values indicate left-skewed (tail risk)

- **Average Excess Kurtosis:** 14.3160
  - Positive values indicate fat tails

- **Normality Tests Passed (Jarque-Bera):** 0/400 portfolios (0.0%)

## Computational Performance

- **Average Runtime per Portfolio:** 1111.05 ms

- **95th Percentile Runtime:** 1261.99 ms

## Key Insights

### Findings

- **Risk Overestimation:** VaR tends to overestimate risk (violation ratio > 1.2)

- **Fat Tails Detected:** Returns exhibit fat tails, which may limit VaR accuracy

### Recommendations

- Consider alternative VaR methods (e.g., Monte Carlo, Historical Simulation) for portfolios with fat-tailed returns
- Monitor portfolios in 'red' traffic light zone more closely
- Adjust confidence levels or horizons based on backtesting results

## Detailed Metrics

### Summary Statistics by Metric

```
       portfolio_id  confidence_level     horizon  estimation_window    hit_rate  violation_ratio  kupiec_unconditional_coverage  kupiec_test_statistic  christoffersen_independence  christoffersen_independence_statistic  christoffersen_conditional_coverage  christoffersen_conditional_coverage_statistic  num_violations  total_observations  expected_violations  mean_exceedance  max_exceedance  std_exceedance  quantile_loss_score  rmse_var_vs_losses  num_active_assets  hhi_concentration  effective_number_of_assets  covariance_condition_number    skewness    kurtosis  jarque_bera_p_value  jarque_bera_statistic  runtime_per_portfolio_ms  p95_runtime_ms  mean_runtime_ms  median_runtime_ms  min_runtime_ms  max_runtime_ms
count     400.00000        400.000000  400.000000              400.0  400.000000       400.000000                          400.0             400.000000                        400.0                             400.000000                                400.0                                     400.000000      400.000000               400.0           400.000000       400.000000      400.000000      400.000000           400.000000          400.000000         400.000000         400.000000                  400.000000                 4.000000e+02  400.000000  400.000000                400.0             400.000000                400.000000      400.000000       400.000000         400.000000      400.000000      400.000000
mean       49.50000          0.970000    5.500000              252.0    0.886392        53.429664                            0.0           14946.101654                          0.0                            1344.631936                                  0.0                                   16290.733590     2190.275000              2471.0            74.130000         0.048713        0.210774        0.019667             0.045995            0.052692           6.040000           0.336270                    3.696964                 1.188948e+01   -0.804314   14.315997                  0.0           22834.763974               1111.047425     1261.989045      1111.047425        1098.674536     1007.244587     1440.284967
std        28.90222          0.020025    4.505636                0.0    0.014707        35.873158                            0.0            3707.574571                          0.0                             287.161519                                  0.0                                    3838.054910       36.342073                 0.0            49.481891         0.027690        0.053664        0.007776             0.026556            0.028465           2.540578           0.178411                    1.560244                 1.778581e-15    0.430952    3.680194                  0.0           11017.609703                  0.000000        0.000000         0.000000           0.000000        0.000000        0.000000
min         0.00000          0.950000    1.000000              252.0    0.853096        17.061918                            0.0           10604.944100                          0.0                             796.734891                                  0.0                                   11401.678991     2108.000000              2471.0            24.710000         0.016341        0.113106        0.009835             0.014859            0.019071           2.000000           0.142291                    1.065007                 1.188948e+01   -1.449692    4.381795                  0.0            1978.370736               1111.047425     1261.989045      1111.047425        1098.674536     1007.244587     1440.284967
25%        24.75000          0.950000    1.000000              252.0    0.877580        17.788345                            0.0           11481.249591                          0.0                            1117.267405                                  0.0                                   12870.995165     2168.500000              2471.0            24.710000         0.023611        0.168810        0.012723             0.022099            0.026965           4.000000           0.209405                    2.367240                 1.188948e+01   -1.110229   11.399792                  0.0           13388.199371               1111.047425     1261.989045      1111.047425        1098.674536     1007.244587     1440.284967
50%        49.50000          0.970000    5.500000              252.0    0.893565        53.035208                            0.0           14978.356862                          0.0                            1440.386136                                  0.0                                   16366.804314     2208.000000              2471.0            74.130000         0.045501        0.206231        0.018193             0.042931            0.049865           6.000000           0.279022                    3.583950                 1.188948e+01   -0.866401   14.655829                  0.0           22377.196540               1111.047425     1261.989045      1111.047425        1098.674536     1007.244587     1440.284967
75%        74.25000          0.990000   10.000000              252.0    0.898422        89.225010                            0.0           18623.036549                          0.0                            1610.373582                                  0.0                                   20030.596098     2220.000000              2471.0           123.550000         0.072930        0.244616        0.025390             0.069260            0.077184           8.000000           0.422513                    4.776810                 1.188948e+01   -0.600521   16.772617                  0.0           29536.058078               1111.047425     1261.989045      1111.047425        1098.674536     1007.244587     1440.284967
max        99.00000          0.990000   10.000000              252.0    0.898422        89.842169                            0.0           18828.370564                          0.0                            1610.373582                                  0.0                                   20438.744146     2220.000000              2471.0           123.550000         0.131177        0.401398        0.049827             0.124562            0.140318          10.000000           0.938961                    7.027832                 1.188948e+01    0.616589   21.416676                  0.0           48089.824486               1111.047425     1261.989045      1111.047425        1098.674536     1007.244587     1440.284967
```
