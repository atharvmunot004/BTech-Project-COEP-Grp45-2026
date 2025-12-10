"""
CVaR Portfolio Optimization Module.

Implements Conditional Value at Risk (CVaR) portfolio optimization using the
Rockafellar-Uryasev linear programming formulation.
"""

from .main import run_cvar_optimization
from .cvar_optimizer import (
    optimize_cvar_portfolio,
    generate_cvar_return_frontier,
    setup_cvar_linear_program
)
from .scenario_generation import (
    generate_scenario_matrix,
    compute_portfolio_scenario_returns
)
from .metrics import (
    compute_var_cvar,
    compute_cvar_sharpe_ratio,
    compute_cvar_sortino_ratio,
    compute_risk_metrics,
    compute_tail_risk_metrics
)
from .returns import (
    load_panel_prices,
    load_baseline_portfolios,
    compute_daily_returns
)

__all__ = [
    'run_cvar_optimization',
    'optimize_cvar_portfolio',
    'generate_cvar_return_frontier',
    'setup_cvar_linear_program',
    'generate_scenario_matrix',
    'compute_portfolio_scenario_returns',
    'compute_var_cvar',
    'compute_cvar_sharpe_ratio',
    'compute_cvar_sortino_ratio',
    'compute_risk_metrics',
    'compute_tail_risk_metrics',
    'load_panel_prices',
    'load_baseline_portfolios',
    'compute_daily_returns'
]

