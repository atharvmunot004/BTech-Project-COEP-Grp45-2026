"""
Black-Litterman Portfolio Optimization Module.

This module implements the Black-Litterman model for portfolio optimization,
combining market equilibrium returns with synthetic investor views to produce posterior
expected returns and optimal portfolio allocations.
"""

from .main import run_black_litterman_optimization
from .black_litterman_optimizer import (
    compute_covariance_matrix,
    derive_market_equilibrium_returns,
    generate_synthetic_views,
    parse_and_scale_views,
    compute_posterior_bl_returns,
    compute_posterior_covariance,
    optimize_portfolio,
    generate_efficient_frontier
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
    compute_information_ratio,
    compute_tracking_error,
    compute_alpha_vs_market,
    compute_bl_specific_metrics,
    compute_structure_metrics,
    compute_risk_metrics,
    compute_distribution_metrics,
    compute_comparison_metrics,
    compute_runtime_metrics
)
from .report_generator import generate_report

__all__ = [
    'run_black_litterman_optimization',
    'compute_covariance_matrix',
    'derive_market_equilibrium_returns',
    'generate_synthetic_views',
    'parse_and_scale_views',
    'compute_posterior_bl_returns',
    'compute_posterior_covariance',
    'optimize_portfolio',
    'generate_efficient_frontier',
    'load_panel_prices',
    'load_baseline_portfolios',
    'compute_daily_returns',
    'compute_portfolio_statistics',
    'compute_sharpe_ratio',
    'compute_sortino_ratio',
    'compute_max_drawdown',
    'compute_calmar_ratio',
    'compute_information_ratio',
    'compute_tracking_error',
    'compute_alpha_vs_market',
    'compute_bl_specific_metrics',
    'compute_structure_metrics',
    'compute_risk_metrics',
    'compute_distribution_metrics',
    'compute_comparison_metrics',
    'compute_runtime_metrics',
    'generate_report'
]

__version__ = '1.0.0'

