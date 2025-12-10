"""
Risk Parity / Equal Risk Contribution (ERC) Portfolio Optimization Module.

This module implements Risk Parity portfolio optimization, which aims to equalize
the risk contribution of each asset in the portfolio. The Equal Risk Contribution (ERC)
approach ensures that each asset contributes equally to the portfolio's total risk.
"""

from .main import run_risk_parity_erc_optimization
from .risk_parity_erc_optimizer import (
    compute_covariance_matrix,
    calculate_risk_contributions,
    optimize_risk_parity_portfolio
)
from .returns import (
    load_panel_prices,
    load_baseline_portfolios,
    compute_daily_returns
)
from .metrics import (
    compute_portfolio_statistics,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_max_drawdown,
    compute_calmar_ratio,
    compute_risk_parity_specific_metrics,
    compute_structure_metrics,
    compute_risk_metrics,
    compute_distribution_metrics,
    compute_comparison_metrics,
    compute_runtime_metrics
)
from .report_generator import generate_report

__all__ = [
    'run_risk_parity_erc_optimization',
    'compute_covariance_matrix',
    'calculate_risk_contributions',
    'optimize_risk_parity_portfolio',
    'load_panel_prices',
    'load_baseline_portfolios',
    'compute_daily_returns',
    'compute_portfolio_statistics',
    'compute_sharpe_ratio',
    'compute_sortino_ratio',
    'compute_max_drawdown',
    'compute_calmar_ratio',
    'compute_risk_parity_specific_metrics',
    'compute_structure_metrics',
    'compute_risk_metrics',
    'compute_distribution_metrics',
    'compute_comparison_metrics',
    'compute_runtime_metrics',
    'generate_report'
]

__version__ = '1.0.0'

