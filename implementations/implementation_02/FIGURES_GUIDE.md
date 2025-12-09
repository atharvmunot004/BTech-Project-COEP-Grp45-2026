# Visualization Guide - Comparative Analysis Figures

## Overview

This document describes all generated visualizations for the comparative analysis. All figures are saved in `results/figures/` directory at 300 DPI resolution, suitable for publication.

---

## Objective 1: Quantum vs Classical Risk Assessment

### 1. Error vs Sample Size Analysis (`error_vs_sample_size.png`)
**Purpose:** Demonstrate O(1/N) vs O(1/√N) scaling for Monte Carlo methods

**Content:**
- Two panels (95% and 99% confidence levels)
- Actual Monte Carlo error vs sample size (log-log scale)
- Theoretical O(1/√N) scaling line
- Theoretical O(1/N) scaling line (for comparison)

**Key Insights:**
- Shows how Monte Carlo error decreases with sample size
- Demonstrates the O(1/√N) convergence rate
- Enables comparison with potential quantum O(1/N) advantage

---

### 2. Runtime Comparison - Risk Methods (`runtime_comparison_risk.png`)
**Purpose:** Compare computational efficiency of different risk assessment methods

**Content:**
- Horizontal bar chart comparing average runtime
- Methods: Monte Carlo, QAE, Parametric VaR, EVT POT
- Error bars showing standard deviation

**Key Insights:**
- Identifies fastest methods for real-time risk assessment
- Shows trade-off between accuracy and speed
- Useful for production system design

---

### 3. VaR/CVaR Comparison (`var_cvar_comparison.png`)
**Purpose:** Compare risk estimates across different methods

**Content:**
- Two panels: CVaR estimates and VaR estimates
- Bar charts with error bars for each method
- Methods: Monte Carlo, QAE, Parametric VaR, EVT POT

**Key Insights:**
- Shows consistency/divergence between methods
- Identifies conservative vs aggressive risk estimates
- Validates method reliability

---

## Objective 2: Quantum-Enhanced Portfolio Optimization

### 4. Portfolio Performance Comparison (`portfolio_performance.png`)
**Purpose:** Comprehensive comparison of portfolio optimization methods

**Content (4-panel figure):**
1. **Backtested Sharpe Ratio:** Bar chart comparing average Sharpe ratios
2. **Risk-Return Tradeoff:** Scatter plot (volatility vs return)
3. **Runtime Comparison:** Horizontal bar chart of optimization time
4. **Solution Sparsity:** Bar chart showing fraction of non-zero weights

**Key Insights:**
- Performance ranking of methods
- Risk-return efficiency frontier
- Computational cost comparison
- Portfolio concentration analysis

---

### 5. Backtest Time Series (`backtest_timeseries.png`)
**Purpose:** Show performance stability across rolling windows

**Content (4-panel figure):**
1. **Sharpe Ratio over Windows:** Line plot across rolling windows
2. **Returns over Windows:** Line plot showing return consistency
3. **Volatility over Windows:** Line plot showing risk stability
4. **Max Drawdown over Windows:** Line plot showing downside risk

**Key Insights:**
- Method robustness over time
- Performance consistency
- Risk stability assessment
- Out-of-sample validation

---

### 6. Scalability Analysis (`scalability_analysis.png`)
**Purpose:** Evaluate method performance as number of assets increases

**Content (2-panel figure):**
1. **Runtime vs Number of Assets:** Line plot showing computational scaling
2. **Sharpe Ratio vs Number of Assets:** Line plot showing performance scaling

**Key Insights:**
- Computational complexity analysis
- Performance degradation with scale
- Practical limits for each method
- Scalability comparison (quantum vs classical)

---

## Comprehensive Summary

### 7. Method Comparison Summary (`method_comparison_summary.png`)
**Purpose:** Single-page overview of all key metrics

**Content (9-panel grid):**
- Risk Methods: CVaR estimates, Runtime
- Portfolio Methods: Sharpe Ratio, Returns, Volatility, Max Drawdown
- Monte Carlo: Runtime vs Sample Size

**Key Insights:**
- Quick reference for all methods
- Publication-ready summary figure
- Comprehensive performance overview

---

## Figure Specifications

- **Resolution:** 300 DPI (publication quality)
- **Format:** PNG
- **Color Scheme:** Consistent palette across all figures
- **Font Sizes:** 10-14pt (readable in publications)
- **Grid:** Enabled for easier reading
- **Legends:** Included for all multi-series plots

---

## Usage Recommendations

### For Research Papers:
1. **Error vs Sample Size:** Use in methodology section to demonstrate scaling
2. **Runtime Comparison:** Use in results section for efficiency analysis
3. **Portfolio Performance:** Use in results section for method comparison
4. **Method Comparison Summary:** Use as main results figure or in supplementary material

### For Presentations:
- All figures are suitable for slides
- Consider enlarging Method Comparison Summary for overview slides
- Use individual panels from multi-panel figures for detailed discussions

### For Reports:
- Use figures in order of objectives
- Reference figure numbers in text
- Include captions describing key findings

---

## Data Sources

All figures are generated from:
- `results/comparative_analysis/risk_comparison.csv` (320 rows)
- `results/comparative_analysis/portfolio_optimization_comparison.csv` (4 rows)
- `results/comparative_analysis/scalability_analysis.csv` (4 rows)
- `results/backtest_results.csv` (20 rows)

---

## Regenerating Figures

To regenerate all figures:
```bash
python generate_visualizations.py
```

Figures will be saved to: `results/figures/`

---

## Customization

The visualization script (`generate_visualizations.py`) can be customized:
- Color schemes (line 12-20)
- Figure sizes (figsize parameters)
- Font sizes (fontsize parameters)
- Style preferences (plt.style.use)

---

*Generated: 2024-12-06*
*Script: `generate_visualizations.py`*


