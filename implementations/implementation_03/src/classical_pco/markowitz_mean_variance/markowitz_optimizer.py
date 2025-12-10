"""
Markowitz Mean-Variance Portfolio Optimizer.

Implements the classical mean-variance optimization framework.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from sklearn.covariance import LedoitWolf
import time
import warnings

# Try to import optimization libraries
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    warnings.warn("cvxpy not available. Using scipy.optimize instead.")

try:
    import osqp
    OSQP_AVAILABLE = True
except ImportError:
    OSQP_AVAILABLE = False

if not CVXPY_AVAILABLE:
    from scipy.optimize import minimize


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
    
    # Select data window
    if window is not None and len(returns) > window:
        returns_window = returns.iloc[-window:]
    else:
        returns_window = returns
    
    # Remove any assets with insufficient data
    returns_window = returns_window.dropna(axis=1, how='any')
    
    if len(returns_window.columns) == 0:
        raise ValueError("No assets with sufficient data for covariance estimation")
    
    # Compute covariance
    if use_shrinkage and shrinkage_method == 'ledoit_wolf':
        # Ledoit-Wolf shrinkage estimator
        lw = LedoitWolf()
        cov_array = lw.fit(returns_window.values).covariance_
        cov_matrix = pd.DataFrame(
            cov_array,
            index=returns_window.columns,
            columns=returns_window.columns
        )
    else:
        # Sample covariance
        cov_matrix = returns_window.cov()
    
    computation_time = (time.time() - start_time) * 1000  # Convert to ms
    
    return cov_matrix, computation_time


def compute_expected_returns(
    returns: pd.DataFrame,
    method: str = 'historical_mean',
    window: Optional[int] = None,
    use_annualization: bool = True
) -> pd.Series:
    """
    Compute expected returns for assets.
    
    Args:
        returns: DataFrame of returns with dates as index and assets as columns
        method: 'historical_mean' for historical mean returns
        window: Optional rolling window size (if None, uses all data)
        use_annualization: Whether to annualize returns (multiply by 252)
        
    Returns:
        Series of expected returns with assets as index
    """
    # Select data window
    if window is not None and len(returns) > window:
        returns_window = returns.iloc[-window:]
    else:
        returns_window = returns
    
    # Remove any assets with insufficient data
    returns_window = returns_window.dropna(axis=1, how='any')
    
    if method == 'historical_mean':
        expected_returns = returns_window.mean()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Annualize if requested
    if use_annualization:
        expected_returns = expected_returns * 252
    
    return expected_returns


def optimize_portfolio(
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    objective: str = 'min_variance',
    risk_aversion: Optional[float] = None,
    risk_free_rate: float = 0.0,
    constraints: Optional[Dict] = None,
    solver: str = 'osqp',
    tolerance: float = 1e-6
) -> Tuple[pd.Series, Dict, float]:
    """
    Optimize portfolio using Markowitz mean-variance framework.
    
    Args:
        expected_returns: Series of expected returns
        covariance_matrix: Covariance matrix DataFrame
        objective: 'min_variance', 'max_sharpe', or 'risk_return_tradeoff'
        risk_aversion: Risk aversion parameter (lambda) for risk-return tradeoff
        risk_free_rate: Risk-free rate for Sharpe ratio calculation
        constraints: Dictionary of constraints
        solver: Solver backend ('osqp', 'cvxpy', 'scipy')
        tolerance: Optimization tolerance
        
    Returns:
        Tuple of (optimal_weights, optimization_info, solver_time_ms)
    """
    start_time = time.time()
    
    # Align assets
    assets = expected_returns.index.intersection(covariance_matrix.index)
    if len(assets) == 0:
        raise ValueError("No common assets between expected returns and covariance matrix")
    
    mu = expected_returns[assets].values
    Sigma = covariance_matrix.loc[assets, assets].values
    n = len(assets)
    
    # Default constraints
    if constraints is None:
        constraints = {
            'long_only': True,
            'fully_invested': True,
            'no_short_selling': True,
            'weight_bounds': [0.0, 1.0],
            'max_weight_per_asset': None,
            'min_num_assets': 1
        }
    
    # Set up optimization problem
    if CVXPY_AVAILABLE and solver in ['cvxpy', 'osqp']:
        w = cp.Variable(n)
        
        # Objective function
        if objective == 'min_variance':
            objective_func = cp.Minimize(cp.quad_form(w, Sigma))
        elif objective == 'max_sharpe':
            # Maximize Sharpe ratio = (mu^T w - rf) / sqrt(w^T Sigma w)
            # Equivalent to minimize: - (mu^T w - rf) / sqrt(w^T Sigma w)
            # Reformulated as: minimize t subject to sqrt(w^T Sigma w) <= t * (mu^T w - rf)
            t = cp.Variable()
            objective_func = cp.Minimize(t)
            # This is a more complex formulation, simplified here
            # For now, use risk-return tradeoff with appropriate lambda
            if risk_aversion is None:
                risk_aversion = 1.0
            objective_func = cp.Minimize(
                -cp.sum(mu @ w) + risk_aversion * cp.quad_form(w, Sigma)
            )
        elif objective == 'risk_return_tradeoff':
            if risk_aversion is None:
                risk_aversion = 1.0
            objective_func = cp.Minimize(
                -cp.sum(mu @ w) + risk_aversion * cp.quad_form(w, Sigma)
            )
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        # Constraints
        constraint_list = []
        
        # Fully invested constraint
        if constraints.get('fully_invested', True):
            constraint_list.append(cp.sum(w) == 1.0)
        
        # Long-only / no short selling
        if constraints.get('long_only', True) or constraints.get('no_short_selling', True):
            constraint_list.append(w >= 0.0)
        
        # Weight bounds
        weight_bounds = constraints.get('weight_bounds', [0.0, 1.0])
        constraint_list.append(w >= weight_bounds[0])
        constraint_list.append(w <= weight_bounds[1])
        
        # Max weight per asset
        max_weight = constraints.get('max_weight_per_asset')
        if max_weight is not None:
            constraint_list.append(w <= max_weight)
        
        # Solve
        problem = cp.Problem(objective_func, constraint_list)
        
        if solver == 'osqp' and OSQP_AVAILABLE:
            problem.solve(solver=cp.OSQP, eps_abs=tolerance, eps_rel=tolerance)
        else:
            problem.solve(solver=cp.ECOS, abstol=tolerance, reltol=tolerance)
        
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            raise RuntimeError(f"Optimization failed with status: {problem.status}")
        
        weights_array = w.value
        solver_time = (time.time() - start_time) * 1000
        
        optimization_info = {
            'status': problem.status,
            'objective_value': problem.value,
            'solver': solver
        }
        
    else:
        # Fallback to scipy.optimize
        if objective == 'min_variance':
            def objective_func(w):
                return w @ Sigma @ w
        elif objective == 'risk_return_tradeoff':
            if risk_aversion is None:
                risk_aversion = 1.0
            def objective_func(w):
                return -mu @ w + risk_aversion * (w @ Sigma @ w)
        else:
            raise ValueError(f"Objective {objective} not supported with scipy solver")
        
        # Constraints
        constraint_list = []
        
        # Fully invested
        if constraints.get('fully_invested', True):
            constraint_list.append({
                'type': 'eq',
                'fun': lambda w: np.sum(w) - 1.0
            })
        
        # Bounds
        weight_bounds = constraints.get('weight_bounds', [0.0, 1.0])
        bounds = [(weight_bounds[0], weight_bounds[1]) for _ in range(n)]
        
        # Max weight per asset
        max_weight = constraints.get('max_weight_per_asset')
        if max_weight is not None:
            for i in range(n):
                bounds[i] = (bounds[i][0], min(bounds[i][1], max_weight))
        
        # Initial guess (equal weights)
        x0 = np.ones(n) / n
        
        # Solve
        result = minimize(
            objective_func,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraint_list,
            options={'ftol': tolerance}
        )
        
        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")
        
        weights_array = result.x
        solver_time = (time.time() - start_time) * 1000
        
        optimization_info = {
            'status': 'success' if result.success else 'failed',
            'objective_value': result.fun,
            'solver': 'scipy'
        }
    
    # Create weights Series
    weights = pd.Series(weights_array, index=assets)
    
    # Normalize to ensure sum is 1 (handle numerical precision)
    weights = weights / weights.sum()
    
    return weights, optimization_info, solver_time


def generate_efficient_frontier(
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    num_portfolios: int = 100,
    risk_levels: Union[str, List[float]] = 'auto',
    constraints: Optional[Dict] = None,
    risk_free_rate: float = 0.0,
    solver: str = 'osqp',
    tolerance: float = 1e-6
) -> pd.DataFrame:
    """
    Generate efficient frontier portfolios.
    
    Args:
        expected_returns: Series of expected returns
        covariance_matrix: Covariance matrix DataFrame
        num_portfolios: Number of portfolios to generate
        risk_levels: 'auto' or list of target risk levels
        constraints: Dictionary of constraints
        risk_free_rate: Risk-free rate
        solver: Solver backend
        tolerance: Optimization tolerance
        
    Returns:
        DataFrame with columns: expected_return, volatility, sharpe_ratio, weights
    """
    # Align assets
    assets = expected_returns.index.intersection(covariance_matrix.index)
    mu = expected_returns[assets].values
    Sigma = covariance_matrix.loc[assets, assets].values
    
    # Determine risk levels
    if risk_levels == 'auto':
        # Find min variance and max return portfolios to determine range
        min_var_weights, _, _ = optimize_portfolio(
            expected_returns[assets],
            covariance_matrix.loc[assets, assets],
            objective='min_variance',
            constraints=constraints,
            solver=solver,
            tolerance=tolerance
        )
        min_var_return = mu @ min_var_weights.values
        min_var_vol = np.sqrt(min_var_weights.values @ Sigma @ min_var_weights.values)
        
        # Max return (subject to constraints)
        max_return = np.max(mu)
        max_vol = np.sqrt(np.max(np.diag(Sigma)))  # Approximate
        
        # Generate risk levels
        target_vols = np.linspace(min_var_vol, max_vol, num_portfolios)
    else:
        target_vols = risk_levels
    
    frontier_portfolios = []
    
    for target_vol in target_vols:
        try:
            # Optimize for target volatility using risk-return tradeoff
            # Approximate lambda to achieve target volatility
            # This is a simplified approach - in practice, use bisection or similar
            lambda_guess = 1.0
            
            weights, _, _ = optimize_portfolio(
                expected_returns[assets],
                covariance_matrix.loc[assets, assets],
                objective='risk_return_tradeoff',
                risk_aversion=lambda_guess,
                constraints=constraints,
                solver=solver,
                tolerance=tolerance
            )
            
            portfolio_return = mu @ weights.values
            portfolio_vol = np.sqrt(weights.values @ Sigma @ weights.values)
            portfolio_sharpe = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else np.nan
            
            frontier_portfolios.append({
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'portfolio_variance': portfolio_vol ** 2,
                'sharpe_ratio': portfolio_sharpe,
                **{f'weight_{asset}': weights[asset] for asset in assets}
            })
        except Exception as e:
            warnings.warn(f"Failed to generate portfolio for target vol {target_vol}: {e}")
            continue
    
    if len(frontier_portfolios) == 0:
        raise RuntimeError("Failed to generate any efficient frontier portfolios")
    
    return pd.DataFrame(frontier_portfolios)

