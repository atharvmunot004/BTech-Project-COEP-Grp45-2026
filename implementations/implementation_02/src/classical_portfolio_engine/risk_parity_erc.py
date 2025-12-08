"""Risk Parity / Equal Risk Contribution."""
import numpy as np
import pandas as pd
from typing import Optional
import time
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Install with: pip install scipy")


class RiskParityERC:
    """Equal Risk Contribution (Risk Parity) Portfolio."""
    
    def __init__(self, long_only: bool = True, sum_to_one: bool = True):
        """
        Initialize Risk Parity optimizer.
        
        Args:
            long_only: Whether to allow only long positions
            sum_to_one: Whether weights must sum to 1
        """
        self.long_only = long_only
        self.sum_to_one = sum_to_one
    
    def optimize(self, covariance_matrix: pd.DataFrame) -> dict:
        """
        Optimize portfolio for equal risk contribution.
        
        Args:
            covariance_matrix: Covariance matrix
            
        Returns:
            Dictionary with ERC weights, risk contributions, runtime stats
        """
        start_time = time.time()
        
        n = len(covariance_matrix)
        Sigma = covariance_matrix.values
        
        # Objective: minimize sum of squared differences in risk contributions
        def objective(w):
            w = w.reshape(-1, 1)
            portfolio_var = (w.T @ Sigma @ w)[0, 0]
            portfolio_std = np.sqrt(portfolio_var)
            
            # Risk contributions
            marginal_contrib = (Sigma @ w).flatten()
            risk_contrib = w.flatten() * marginal_contrib / portfolio_std
            
            # Target: equal risk contributions
            target_contrib = portfolio_std / n
            
            # Minimize squared differences
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        # Constraints
        constraints = []
        if self.sum_to_one:
            constraints.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        
        bounds = []
        for i in range(n):
            if self.long_only:
                bounds.append((0, 1))
            else:
                bounds.append((-1, 1))
        
        # Initial guess: equal weights
        w0 = np.ones(n) / n
        
        # Optimize
        if SCIPY_AVAILABLE:
            result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                optimal_weights = result.x
            else:
                # Fallback: equal weights
                optimal_weights = w0
        else:
            # Fallback: equal weights
            optimal_weights = w0
        
        # Calculate risk contributions
        w_vec = optimal_weights.reshape(-1, 1)
        portfolio_var = (w_vec.T @ Sigma @ w_vec)[0, 0]
        portfolio_std = np.sqrt(portfolio_var)
        marginal_contrib = (Sigma @ w_vec).flatten()
        risk_contrib = optimal_weights * marginal_contrib / portfolio_std
        
        runtime = time.time() - start_time
        
        return {
            'erc_weights': optimal_weights,
            'risk_contributions': risk_contrib,
            'runtime_stats': {
                'wall_clock_time': runtime,
                'method': 'risk_parity_erc'
            },
            'diagnostics': {
                'portfolio_volatility': portfolio_std,
                'risk_concentration_index': np.std(risk_contrib) / np.mean(risk_contrib) if np.mean(risk_contrib) > 0 else 0
            }
        }

