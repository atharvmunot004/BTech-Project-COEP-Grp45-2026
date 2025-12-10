"""
Markowitz Mean-Variance Portfolio Optimization Module.

Implements classical mean-variance portfolio optimization following Markowitz (1952).
"""

from .main import run_markowitz_optimization
from .markowitz_optimizer import (
    compute_covariance_matrix,
    compute_expected_returns,
    optimize_portfolio,
    generate_efficient_frontier
)

__all__ = [
    'run_markowitz_optimization',
    'compute_covariance_matrix',
    'compute_expected_returns',
    'optimize_portfolio',
    'generate_efficient_frontier'
]

