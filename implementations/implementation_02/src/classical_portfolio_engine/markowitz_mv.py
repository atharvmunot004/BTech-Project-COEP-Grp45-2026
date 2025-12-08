"""Markowitz Mean-Variance Optimization."""
import numpy as np
import pandas as pd
from typing import Optional
import time
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    print("Warning: cvxpy not available. Install with: pip install cvxpy")


class MarkowitzMV:
    """Markowitz Mean-Variance Portfolio Optimization."""
    
    def __init__(self, risk_aversion: float = 1.0, long_only: bool = True, sum_to_one: bool = True):
        """
        Initialize Markowitz optimizer.
        
        Args:
            risk_aversion: Risk aversion parameter
            long_only: Whether to allow only long positions
            sum_to_one: Whether weights must sum to 1
        """
        self.risk_aversion = risk_aversion
        self.long_only = long_only
        self.sum_to_one = sum_to_one
        if not CVXPY_AVAILABLE:
            print("Warning: cvxpy not available. Will use fallback optimization.")
    
    def optimize(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_aversion: Optional[float] = None
    ) -> dict:
        """
        Optimize portfolio weights using Markowitz mean-variance.
        
        Args:
            expected_returns: Expected returns per asset
            covariance_matrix: Covariance matrix
            risk_aversion: Risk aversion parameter (overrides init value)
            
        Returns:
            Dictionary with optimal weights, efficient frontier points, runtime stats
        """
        start_time = time.time()
        
        if risk_aversion is None:
            risk_aversion = self.risk_aversion
        
        n = len(expected_returns)
        mu = expected_returns.values
        Sigma = covariance_matrix.values
        
        if CVXPY_AVAILABLE:
            # Define optimization problem
            w = cp.Variable(n)
            
            # Objective: maximize return - risk_aversion * variance
            objective = cp.Maximize(mu @ w - risk_aversion * cp.quad_form(w, Sigma))
            
            # Constraints
            constraints = []
            if self.long_only:
                constraints.append(w >= 0)
            if self.sum_to_one:
                constraints.append(cp.sum(w) == 1)
            
            # Solve
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            if problem.status not in ["infeasible", "unbounded"]:
                optimal_weights = w.value
            else:
                # Fallback: equal weights
                optimal_weights = np.ones(n) / n
        else:
            # Fallback: simple optimization without cvxpy
            # Use equal weights as baseline
            optimal_weights = np.ones(n) / n
        
        # Calculate efficient frontier
        frontier_returns = []
        frontier_risks = []
        if CVXPY_AVAILABLE:
            for target_return in np.linspace(mu.min(), mu.max(), 50):
                w_frontier = cp.Variable(n)
                objective_frontier = cp.Minimize(cp.quad_form(w_frontier, Sigma))
                constraints_frontier = [mu @ w_frontier >= target_return]
                if self.long_only:
                    constraints_frontier.append(w_frontier >= 0)
                if self.sum_to_one:
                    constraints_frontier.append(cp.sum(w_frontier) == 1)
                
                problem_frontier = cp.Problem(objective_frontier, constraints_frontier)
                problem_frontier.solve()
                
                if problem_frontier.status not in ["infeasible", "unbounded"]:
                    w_val = w_frontier.value
                    frontier_returns.append(mu @ w_val)
                    frontier_risks.append(np.sqrt(w_val @ Sigma @ w_val))
        else:
            # Simplified frontier without cvxpy
            frontier_returns = [mu @ optimal_weights]
            frontier_risks = [np.sqrt(optimal_weights @ Sigma @ optimal_weights)]
        
        runtime = time.time() - start_time
        
        return {
            'optimal_weights_mv': optimal_weights,
            'efficient_frontier_points': {
                'returns': frontier_returns,
                'risks': frontier_risks
            },
            'runtime_stats': {
                'wall_clock_time': runtime,
                'method': 'markowitz_mv'
            },
            'diagnostics': {
                'expected_return': mu @ optimal_weights,
                'volatility': np.sqrt(optimal_weights @ Sigma @ optimal_weights),
                'sharpe_ratio': (mu @ optimal_weights) / np.sqrt(optimal_weights @ Sigma @ optimal_weights) if np.sqrt(optimal_weights @ Sigma @ optimal_weights) > 0 else 0
            }
        }

