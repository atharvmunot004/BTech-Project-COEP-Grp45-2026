"""
Variance-Covariance Value-at-Risk (Parametric VaR) Evaluation Module.

This module implements comprehensive VaR evaluation including:
- Daily and portfolio returns computation
- Rolling VaR calculation
- Backtesting with violation analysis
- Comprehensive performance metrics
- Report generation
"""

from .main import evaluate_var
from .returns import compute_daily_returns, compute_portfolio_returns
from .var_calculator import compute_rolling_var
from .backtesting import detect_var_violations, compute_accuracy_metrics
from .metrics import (
    compute_tail_metrics,
    compute_structure_metrics,
    compute_distribution_metrics
)
from .time_sliced_metrics import compute_time_sliced_metrics

__all__ = [
    'evaluate_var',
    'compute_daily_returns',
    'compute_portfolio_returns',
    'compute_rolling_var',
    'detect_var_violations',
    'compute_accuracy_metrics',
    'compute_tail_metrics',
    'compute_structure_metrics',
    'compute_distribution_metrics',
    'compute_time_sliced_metrics',
]

