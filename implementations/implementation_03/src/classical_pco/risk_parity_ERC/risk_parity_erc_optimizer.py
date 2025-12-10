"""
Risk Parity / Equal Risk Contribution (ERC) Portfolio Optimizer.

Implements the Equal Risk Contribution portfolio optimization method, which aims
to equalize the risk contribution of each asset in the portfolio.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.covariance import LedoitWolf
import time
import warnings

# Try to import optimization libraries
try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available. Risk Parity optimization may not work.")


def compute_covariance_matrix(
    returns: pd.DataFrame,
    method: str = 'sample',
    window: Optional[int] = None,
    use_shrinkage: bool = False,
    shrinkage_method: str = 'ledoit_wolf'
) -> Tuple[pd.DataFrame, float]:
    """
    Compute covariance matrix from returns.
    
    Args:
        returns: DataFrame of returns with dates as index and assets as columns
        method: 'sample' for sample covariance
        window: Optional rolling window size (if None, uses all data)
        use_shrinkage: Whether to apply shrinkage estimator
        shrinkage_method: Shrinkage method ('ledoit_wolf')
        
    Returns:
        Tuple of (covariance_matrix, computation_time_ms)
    """
    start_time = time.time()
    
    if window is not None and len(returns) > window:
        returns_window = returns.iloc[-window:]
    else:
        returns_window = returns
    
    returns_window = returns_window.dropna(axis=1, how='any')
    
    if len(returns_window.columns) == 0:
        raise ValueError("No assets with sufficient data for covariance estimation")
    
    if use_shrinkage and shrinkage_method == 'ledoit_wolf':
        lw = LedoitWolf()
        cov_array = lw.fit(returns_window.values).covariance_
        cov_matrix = pd.DataFrame(
            cov_array,
            index=returns_window.columns,
            columns=returns_window.columns
        )
    else:
        cov_matrix = returns_window.cov()
    
    computation_time = (time.time() - start_time) * 1000
    
    return cov_matrix, computation_time


def calculate_risk_contributions(
    weights: np.ndarray,
    covariance_matrix: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate risk contributions for each asset in the portfolio.
    
    The risk contribution of asset i is:
    RC_i = w_i * (Σ * w)_i / σ_p
    
    where:
    - w_i: weight of asset i
    - (Σ * w)_i: marginal contribution to risk
    - σ_p: portfolio volatility
    
    Args:
        weights: Portfolio weights array of shape (n_assets,)
        covariance_matrix: Covariance matrix DataFrame
        
    Returns:
        Tuple of (risk_contributions, marginal_risk_contributions, portfolio_volatility)
    """
    weights = weights.reshape(-1, 1)
    Sigma = covariance_matrix.values
    
    # Portfolio variance
    portfolio_variance = (weights.T @ Sigma @ weights)[0, 0]
    portfolio_volatility = np.sqrt(portfolio_variance)
    
    if portfolio_volatility == 0:
        return np.zeros(len(weights)), np.zeros(len(weights)), 0.0
    
    # Marginal risk contributions: Σ * w
    marginal_contrib = (Sigma @ weights).flatten()
    
    # Risk contributions: w_i * (Σ * w)_i / σ_p
    risk_contrib = (weights.flatten() * marginal_contrib) / portfolio_volatility
    
    return risk_contrib, marginal_contrib, portfolio_volatility


def optimize_risk_parity_portfolio(
    covariance_matrix: pd.DataFrame,
    constraints: Dict,
    optimization_settings: Dict,
    initial_weights: Optional[np.ndarray] = None
) -> Tuple[pd.Series, Dict, float]:
    """
    Optimize portfolio for Equal Risk Contribution (ERC).
    
    The optimization problem minimizes the sum of squared differences between
    each asset's risk contribution and the target equal risk contribution.
    
    Objective: minimize sum((RC_i - target_RC)^2)
    
    where target_RC = σ_p / n (equal risk contribution per asset)
    
    Args:
        covariance_matrix: Covariance matrix DataFrame
        constraints: Dictionary of portfolio constraints
        optimization_settings: Dictionary of optimization solver settings
        initial_weights: Optional initial weights guess (default: equal weights)
        
    Returns:
        Tuple of (optimal_weights, optimization_info, solver_time_ms)
    """
    start_time = time.time()
    
    if not SCIPY_AVAILABLE:
        raise RuntimeError("scipy.optimize is required for Risk Parity optimization")
    
    n = len(covariance_matrix)
    Sigma = covariance_matrix.values
    
    # Initial guess: equal weights
    if initial_weights is None:
        w0 = np.ones(n) / n
    else:
        w0 = initial_weights.copy()
        if len(w0) != n:
            raise ValueError(f"Initial weights length {len(w0)} does not match assets {n}")
    
    # Objective function: minimize sum of squared differences in risk contributions
    def objective(w):
        w = w.reshape(-1, 1)
        portfolio_var = (w.T @ Sigma @ w)[0, 0]
        portfolio_std = np.sqrt(portfolio_var)
        
        if portfolio_std == 0:
            return 1e10  # Penalty for zero volatility
        
        # Risk contributions
        marginal_contrib = (Sigma @ w).flatten()
        risk_contrib = w.flatten() * marginal_contrib / portfolio_std
        
        # Target: equal risk contributions
        target_contrib = portfolio_std / n
        
        # Minimize squared differences
        return np.sum((risk_contrib - target_contrib) ** 2)
    
    # Constraints
    constraint_list = []
    
    # Fully invested constraint: sum(w) = 1
    if constraints.get('fully_invested', True):
        constraint_list.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0
        })
    
    # Bounds
    weight_bounds = constraints.get('weight_bounds', [0.0, 1.0])
    min_weight = weight_bounds[0]
    max_weight = weight_bounds[1]
    
    # Long-only constraint
    if constraints.get('long_only', True) or constraints.get('no_short_selling', False):
        min_weight = max(0.0, min_weight)
    
    # Max weight per asset constraint
    max_weight_per_asset = constraints.get('max_weight_per_asset', 1.0)
    max_weight = min(max_weight, max_weight_per_asset)
    
    bounds = [(min_weight, max_weight) for _ in range(n)]
    
    # Solver settings
    solver_method = optimization_settings.get('backend', 'scipy_slsqp')
    if solver_method == 'scipy_slsqp':
        method = 'SLSQP'
    elif solver_method == 'scipy_lbfgsb':
        method = 'L-BFGS-B'
    else:
        method = 'SLSQP'
    
    tolerance = optimization_settings.get('tolerance', 1e-8)
    max_iter = optimization_settings.get('max_iterations', 5000)
    
    # Optimize
    try:
        result = minimize(
            objective,
            w0,
            method=method,
            bounds=bounds,
            constraints=constraint_list,
            options={
                'ftol': tolerance,
                'maxiter': max_iter,
                'disp': False
            }
        )
        
        if result.success:
            optimal_weights = result.x
            # Ensure weights sum to 1 (numerical precision)
            optimal_weights = optimal_weights / np.sum(optimal_weights)
            # Clip to bounds
            optimal_weights = np.clip(optimal_weights, min_weight, max_weight)
            # Renormalize
            optimal_weights = optimal_weights / np.sum(optimal_weights)
        else:
            warnings.warn(f"Optimization did not converge: {result.message}. Using equal weights.")
            optimal_weights = np.ones(n) / n
        
    except Exception as e:
        warnings.warn(f"Optimization failed: {e}. Using equal weights.")
        optimal_weights = np.ones(n) / n
        result = type('obj', (object,), {'success': False, 'message': str(e)})()
    
    solver_time = (time.time() - start_time) * 1000
    
    # Create weights Series
    optimal_weights_series = pd.Series(optimal_weights, index=covariance_matrix.index)
    
    # Optimization info
    opt_info = {
        'status': 'success' if result.success else 'failed',
        'message': getattr(result, 'message', 'Unknown'),
        'iterations': getattr(result, 'nit', None),
        'objective_value': getattr(result, 'fun', None)
    }
    
    return optimal_weights_series, opt_info, solver_time

