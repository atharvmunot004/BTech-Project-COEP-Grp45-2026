"""Black-Litterman Model."""
import numpy as np
import pandas as pd
from typing import Optional, Dict
import time
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False


class BlackLitterman:
    """Black-Litterman Portfolio Optimization."""
    
    def __init__(self, risk_aversion: float = 3.0, tau: float = 0.05):
        """
        Initialize Black-Litterman optimizer.
        
        Args:
            risk_aversion: Risk aversion parameter
            tau: Scaling factor for uncertainty in prior
        """
        if not CVXPY_AVAILABLE:
            raise ImportError("cvxpy required")
        self.risk_aversion = risk_aversion
        self.tau = tau
    
    def optimize(
        self,
        prior_returns: pd.Series,
        market_caps_or_benchmark_weights: np.ndarray,
        covariance_matrix: pd.DataFrame,
        investor_views: Optional[Dict] = None
    ) -> dict:
        """
        Optimize portfolio using Black-Litterman model.
        
        Args:
            prior_returns: Prior expected returns
            market_caps_or_benchmark_weights: Market capitalization or benchmark weights
            covariance_matrix: Covariance matrix
            investor_views: Dictionary with investor views (optional)
            
        Returns:
            Dictionary with BL implied returns, optimal weights, runtime stats
        """
        start_time = time.time()
        
        n = len(prior_returns)
        mu_prior = prior_returns.values
        Sigma = covariance_matrix.values
        
        # Market-implied returns (reverse optimization)
        w_market = market_caps_or_benchmark_weights / market_caps_or_benchmark_weights.sum()
        pi_market = self.risk_aversion * Sigma @ w_market
        
        # BL posterior returns
        if investor_views is None:
            # No views: use market-implied returns
            mu_bl = pi_market
        else:
            # With views: combine prior and views
            # Simplified implementation
            mu_bl = mu_prior * (1 - self.tau) + pi_market * self.tau
        
        # Optimize using Markowitz with BL returns
        w = cp.Variable(n)
        objective = cp.Maximize(mu_bl @ w - self.risk_aversion * cp.quad_form(w, Sigma))
        constraints = [w >= 0, cp.sum(w) == 1]
        
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status not in ["infeasible", "unbounded"]:
            optimal_weights = w.value
        else:
            optimal_weights = w_market
        
        runtime = time.time() - start_time
        
        return {
            'bl_implied_returns': pd.Series(mu_bl, index=prior_returns.index),
            'optimal_weights_bl': optimal_weights,
            'runtime_stats': {
                'wall_clock_time': runtime,
                'method': 'black_litterman'
            },
            'diagnostics': {
                'expected_return': mu_bl @ optimal_weights,
                'volatility': np.sqrt(optimal_weights @ Sigma @ optimal_weights)
            }
        }

