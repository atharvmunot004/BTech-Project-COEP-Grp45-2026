"""CVaR Optimization."""
import numpy as np
import pandas as pd
from typing import Optional
import time
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False


class CVaROptimization:
    """CVaR-based Portfolio Optimization."""
    
    def __init__(self, confidence_level: float = 0.95, risk_aversion: float = 1.0):
        """
        Initialize CVaR optimizer.
        
        Args:
            confidence_level: Confidence level for CVaR (e.g., 0.95)
            risk_aversion: Risk aversion parameter
        """
        if not CVXPY_AVAILABLE:
            raise ImportError("cvxpy required")
        self.confidence_level = confidence_level
        self.risk_aversion = risk_aversion
    
    def optimize(
        self,
        scenario_return_matrix: np.ndarray,
        confidence_level: Optional[float] = None,
        risk_aversion: Optional[float] = None
    ) -> dict:
        """
        Optimize portfolio to minimize CVaR.
        
        Args:
            scenario_return_matrix: Matrix of scenario returns (scenarios x assets)
            confidence_level: Confidence level (overrides init value)
            risk_aversion: Risk aversion parameter (overrides init value)
            
        Returns:
            Dictionary with optimal weights, CVaR value, runtime stats
        """
        start_time = time.time()
        
        if confidence_level is None:
            confidence_level = self.confidence_level
        if risk_aversion is None:
            risk_aversion = self.risk_aversion
        
        n_scenarios, n_assets = scenario_return_matrix.shape
        alpha = 1 - confidence_level
        
        # CVaR optimization using linear programming formulation
        w = cp.Variable(n_assets)
        VaR = cp.Variable()
        u = cp.Variable(n_scenarios)  # Auxiliary variables for excess losses
        
        # Objective: minimize CVaR (approximated as VaR + mean of excesses)
        objective = cp.Minimize(VaR + (1 / (alpha * n_scenarios)) * cp.sum(u))
        
        # Constraints
        constraints = [
            w >= 0,  # Long only
            cp.sum(w) == 1,  # Fully invested
            u >= 0,  # Excess losses are non-negative
            u >= -scenario_return_matrix @ w - VaR  # Excess loss definition
        ]
        
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status not in ["infeasible", "unbounded"]:
            optimal_weights = w.value
            cvar_value = VaR.value + (1 / (alpha * n_scenarios)) * np.sum(u.value)
        else:
            # Fallback: equal weights
            optimal_weights = np.ones(n_assets) / n_assets
            portfolio_returns = scenario_return_matrix @ optimal_weights
            var = np.percentile(portfolio_returns, alpha * 100)
            tail_losses = portfolio_returns[portfolio_returns <= var]
            cvar_value = -np.mean(tail_losses) if len(tail_losses) > 0 else -var
        
        runtime = time.time() - start_time
        
        return {
            'optimal_weights_cvar': optimal_weights,
            'cvar_value': cvar_value,
            'runtime_stats': {
                'wall_clock_time': runtime,
                'method': 'cvar_optimization'
            },
            'diagnostics': {
                'expected_return': np.mean(scenario_return_matrix @ optimal_weights),
                'volatility': np.std(scenario_return_matrix @ optimal_weights)
            }
        }

