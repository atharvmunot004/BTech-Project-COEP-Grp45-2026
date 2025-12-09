"""
Monte Carlo Simulation for Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR) Evaluation Module.

This module implements comprehensive VaR and CVaR evaluation using Monte Carlo simulation including:
- Daily and portfolio returns computation
- Rolling VaR and CVaR calculation via Monte Carlo simulation
- Backtesting with violation analysis for both VaR and CVaR
- Comprehensive performance metrics
- Report generation
"""

from .main import evaluate_monte_carlo_var_cvar
from .returns import compute_daily_returns, compute_portfolio_returns
from .monte_carlo_calculator import compute_rolling_var, compute_rolling_cvar, align_returns_and_var
from .backtesting import detect_var_violations, detect_cvar_violations, compute_accuracy_metrics
from .metrics import (
    compute_tail_metrics,
    compute_cvar_tail_metrics,
    compute_structure_metrics,
    compute_distribution_metrics
)
from .time_sliced_metrics import compute_time_sliced_metrics

__all__ = [
    'evaluate_monte_carlo_var_cvar',
    'compute_daily_returns',
    'compute_portfolio_returns',
    'compute_rolling_var',
    'compute_rolling_cvar',
    'detect_var_violations',
    'detect_cvar_violations',
    'compute_accuracy_metrics',
    'compute_tail_metrics',
    'compute_cvar_tail_metrics',
    'compute_structure_metrics',
    'compute_distribution_metrics',
    'compute_time_sliced_metrics',
]

