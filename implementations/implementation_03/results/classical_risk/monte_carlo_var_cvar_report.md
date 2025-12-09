# Monte Carlo Simulation for VaR/CVaR Evaluation Report

**Generated:** 2025-12-09 20:42:01

## Methodology Overview

### Monte Carlo Simulation for VaR and CVaR

Monte Carlo simulation generates multiple scenarios of portfolio returns based on historical data.
VaR and CVaR are then calculated from the distribution of simulated returns.

**Advantages:**
- Can capture non-normal distributions
- Flexible with different distribution assumptions
- Provides both VaR and CVaR estimates

**Method:**
1. Estimate mean and covariance from historical returns
2. Simulate multiple scenarios of future returns
3. Calculate VaR as the quantile of simulated losses
4. Calculate CVaR as the expected loss beyond VaR

## Monte Carlo Simulation Details

### Simulation Parameters

- **Number of Simulations:** 10,000
- **Distribution Type:** multivariate_normal
- **Random Seed:** 42
- **Confidence Levels:** [0.95, 0.99]
- **Horizons:** [1, 10] days
- **Estimation Windows:** [252] days

## Summary Statistics

- **Total Portfolios Evaluated:** 400

## Backtesting Results

- **Average Hit Rate:** 0.0135

- **Average Violation Ratio:** 0.5581
  - Ratio > 1 indicates overestimation of risk
  - Ratio < 1 indicates underestimation of risk

### Traffic Light Zones

- **Green:** 385 portfolios (96.2%)
- **Yellow:** 15 portfolios (3.8%)

- **Kupiec Test Passed:** 66/400 portfolios (16.5%)

## VaR vs CVaR Comparison

CVaR (Conditional Value at Risk) provides additional insight into tail risk beyond VaR.

- **Average CVaR Mean Exceedance:** 0.015040
- **Average VaR Mean Exceedance:** 0.014513
- **Difference:** 0.000528

- **Average CVaR Max Exceedance:** 0.066017
- **Average VaR Max Exceedance:** 0.069926

## Tail Risk Analysis

- **Average Mean Exceedance (VaR):** 0.014513

- **Average Max Exceedance (VaR):** 0.069926

- **Average CVaR Mean Exceedance:** 0.015040

- **Average CVaR Max Exceedance:** 0.066017

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

- **Average Simulation Time per Portfolio:** 261589.01 ms

- **Average Total Runtime per Portfolio:** 1046356.06 ms

- **95th Percentile Runtime:** 1255371.68 ms

## Key Insights

### Findings

- **Risk Underestimation:** VaR tends to underestimate risk (violation ratio < 0.8)


- **Fat Tails Detected:** Returns exhibit fat tails, which Monte Carlo simulation can better capture than parametric methods

### Recommendations

- Use CVaR for portfolios with significant tail risk (when CVaR >> VaR)
- Consider increasing number of simulations for more stable estimates
- Monitor portfolios in 'red' traffic light zone more closely
- Adjust confidence levels or horizons based on backtesting results

## Detailed Metrics

### Summary Statistics by Metric

```
       portfolio_id  confidence_level     horizon  estimation_window  var_runtime_ms  simulation_time_ms    hit_rate  violation_ratio  kupiec_unconditional_coverage  kupiec_test_statistic  christoffersen_independence  christoffersen_independence_statistic  christoffersen_conditional_coverage  christoffersen_conditional_coverage_statistic  num_violations  total_observations  expected_violations  mean_exceedance  max_exceedance  std_exceedance  quantile_loss_score  rmse_var_vs_losses  cvar_mean_exceedance  cvar_max_exceedance  cvar_std_exceedance  portfolio_size  num_active_assets  hhi_concentration  effective_number_of_assets  covariance_condition_number    skewness    kurtosis  jarque_bera_p_value  jarque_bera_statistic  runtime_per_portfolio_ms  p95_runtime_ms  mean_runtime_ms  median_runtime_ms  min_runtime_ms  max_runtime_ms
count     400.00000        400.000000  400.000000              400.0      400.000000          400.000000  400.000000       400.000000                     386.000000             386.000000                 3.860000e+02                             386.000000                         3.860000e+02                                     386.000000      400.000000               400.0           400.000000       386.000000      386.000000      325.000000           386.000000          386.000000            344.000000           344.000000           275.000000      400.000000         400.000000         400.000000                  400.000000                 4.000000e+02  400.000000  400.000000                400.0             400.000000              4.000000e+02    4.000000e+02     4.000000e+02       4.000000e+02      400.000000    4.000000e+02
mean       49.50000          0.970000    5.500000              252.0   261589.014353       261589.014353    0.013540         0.558134                       0.050448              67.226011                 4.820249e-01                               4.042619                         1.942329e-02                                      71.268630       33.457500              2471.0            74.130000         0.014513        0.069926        0.017537             0.001389            0.021234              0.015040             0.066017             0.017938        6.040000           6.040000           0.336270                    3.696964                 1.188948e+01   -0.804314   14.315997                  0.0           22834.763974              1.046356e+06    1.255372e+06     1.046356e+06       1.033267e+06   412310.291529    1.257012e+06
std        28.90222          0.020025    4.505636                0.0   204420.037459       204420.037459    0.015100         0.574002                       0.143899              87.093698                 4.402231e-01                               5.538034                         7.208215e-02                                      84.528784       37.311574                 0.0            49.481891         0.006808        0.042527        0.006411             0.000926            0.007963              0.008441             0.042914             0.006222        2.540578           2.540578           0.178411                    1.560244                 3.557163e-15    0.430952    3.680194                  0.0           11017.609703              0.000000e+00    0.000000e+00     0.000000e+00       0.000000e+00        0.000000    0.000000e+00
min         0.00000          0.950000    1.000000              252.0    24604.095221        24604.095221    0.000000         0.000000                       0.000000               0.003425                 3.144869e-07                               0.000810                         0.000000e+00                                       1.247433        0.000000              2471.0            24.710000         0.001298        0.001542        0.001708             0.000386            0.001542              0.000493             0.000493             0.000960        2.000000           2.000000           0.142291                    1.065007                 1.188948e+01   -1.449692    4.381795                  0.0            1978.370736              1.046356e+06    1.255372e+06     1.046356e+06       1.033267e+06   412310.291529    1.257012e+06
25%        24.75000          0.950000    1.000000              252.0    58620.214343        58620.214343    0.001113         0.040469                       0.000000               5.493693                 6.829605e-03                               0.007296                         0.000000e+00                                      13.640815        2.750000              2471.0            24.710000         0.009984        0.030968        0.013782             0.000644            0.016636              0.010727             0.023144             0.014442        4.000000           4.000000           0.209405                    2.367240                 1.188948e+01   -1.110229   11.399792                  0.0           13388.199371              1.046356e+06    1.255372e+06     1.046356e+06       1.033267e+06   412310.291529    1.257012e+06
50%        49.50000          0.970000    5.500000              252.0   155740.576625       155740.576625    0.006070         0.331849                       0.000045              16.765404                 4.631134e-01                               0.541878                         1.885483e-07                                      30.967823       15.000000              2471.0            74.130000         0.013112        0.071403        0.017186             0.001177            0.021030              0.013799             0.074409             0.019063        6.000000           6.000000           0.279022                    3.583950                 1.188948e+01   -0.866401   14.655829                  0.0           22377.196540              1.046356e+06    1.255372e+06     1.046356e+06       1.033267e+06   412310.291529    1.257012e+06
75%        74.25000          0.990000   10.000000              252.0   459686.223209       459686.223209    0.021044         0.878187                       0.019085             192.901426                 9.319287e-01                               7.339607                         1.092452e-03                                     192.953417       52.000000              2471.0           123.550000         0.016983        0.104178        0.020375             0.002248            0.025505              0.016817             0.100640             0.021857        8.000000           8.000000           0.422513                    4.776810                 1.188948e+01   -0.600521   16.772617                  0.0           29536.058078              1.046356e+06    1.255372e+06     1.046356e+06       1.033267e+06   412310.291529    1.257012e+06
max        99.00000          0.990000   10.000000              252.0   575819.443941       575819.443941    0.043707         1.902064                       0.953334             232.793912                 9.772942e-01                              26.158640                         5.359488e-01                                     232.797153      108.000000              2471.0           123.550000         0.044545        0.179341        0.054322             0.004285            0.058819              0.053776             0.168640             0.034204       10.000000          10.000000           0.938961                    7.027832                 1.188948e+01    0.616589   21.416676                  0.0           48089.824486              1.046356e+06    1.255372e+06     1.046356e+06       1.033267e+06   412310.291529    1.257012e+06
```
