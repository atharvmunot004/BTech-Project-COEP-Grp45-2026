"""
GARCH(1,1) Volatility Forecasting for VaR/CVaR Evaluation.

This module provides functionality for evaluating Value at Risk (VaR) and
Conditional Value at Risk (CVaR) using GARCH(1,1) volatility forecasting.
"""

from .main import evaluate_garch_var_cvar
from .returns import compute_daily_returns, compute_portfolio_returns
from .garch_calculator import (
    fit_garch_model,
    compute_conditional_volatility,
    compute_rolling_volatility_forecast,
    compute_rolling_var_from_garch,
    compute_rolling_cvar_from_garch,
    var_from_volatility,
    cvar_from_volatility
)
from .backtesting import (
    detect_var_violations,
    detect_cvar_violations,
    compute_accuracy_metrics
)
from .metrics import (
    compute_tail_metrics,
    compute_cvar_tail_metrics,
    compute_structure_metrics,
    compute_distribution_metrics,
    compute_runtime_metrics
)
from .time_sliced_metrics import compute_time_sliced_metrics

__all__ = [
    'evaluate_garch_var_cvar',
    'compute_daily_returns',
    'compute_portfolio_returns',
    'fit_garch_model',
    'compute_conditional_volatility',
    'compute_rolling_volatility_forecast',
    'compute_rolling_var_from_garch',
    'compute_rolling_cvar_from_garch',
    'var_from_volatility',
    'cvar_from_volatility',
    'detect_var_violations',
    'detect_cvar_violations',
    'compute_accuracy_metrics',
    'compute_tail_metrics',
    'compute_cvar_tail_metrics',
    'compute_structure_metrics',
    'compute_distribution_metrics',
    'compute_runtime_metrics',
    'compute_time_sliced_metrics'
]

