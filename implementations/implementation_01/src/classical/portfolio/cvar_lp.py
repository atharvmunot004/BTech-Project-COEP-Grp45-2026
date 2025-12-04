"""
CVaR-based Portfolio Optimization using Linear Programming.
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict
import cvxpy as cp


def cvar_optimization(
    returns: np.ndarray,
    confidence_level: float = 0.95,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    target_return: Optional[float] = None
) -> Dict:
    """
    Optimize portfolio to minimize CVaR.
    
    This uses the linear programming formulation of CVaR optimization.
    
    Args:
        returns: Historical returns matrix (T x N) where T is time periods, N is assets
        confidence_level: Confidence level for CVaR (e.g., 0.95)
        min_weight: Minimum weight per asset
        max_weight: Maximum weight per asset
        target_return: Optional target return constraint
        
    Returns:
        Dictionary with optimal weights and CVaR
    """
    T, n_assets = returns.shape
    alpha = 1 - confidence_level
    
    # Calculate bounds for VaR based on historical returns
    # VaR should be between min and max of possible portfolio returns
    mean_returns = np.mean(returns, axis=0)
    min_possible_return = np.min(returns @ np.ones(n_assets) / n_assets)  # Equal weights min
    max_possible_return = np.max(returns @ np.ones(n_assets) / n_assets)  # Equal weights max
    # VaR will be negative of a return, so bounds should reflect that
    var_lower = -max_possible_return * 2  # Allow some margin
    var_upper = -min_possible_return * 0.5  # Allow some margin
    
    # Decision variables
    w = cp.Variable(n_assets)  # Portfolio weights
    z = cp.Variable(T, nonneg=True)  # Auxiliary variables for CVaR (non-negative)
    var = cp.Variable()  # VaR threshold
    
    # Portfolio returns for each scenario
    portfolio_returns = returns @ w
    
    # CVaR formulation: minimize CVaR = var + (1/alpha) * mean(max(0, var - portfolio_return))
    # This is equivalent to: minimize var + (1/(alpha*T)) * sum(z)
    # where z >= var - portfolio_return, z >= 0
    # Note: We minimize the negative of returns (losses), so we need to adjust the formulation
    
    # For CVaR, we want to minimize: var + (1/(alpha*T)) * sum(z)
    # where z represents the excess loss beyond VaR
    objective = cp.Minimize(var + (1 / (alpha * T)) * cp.sum(z))
    
    # Constraints
    constraints = [
        cp.sum(w) == 1,  # Budget constraint
        w >= min_weight,
        w <= max_weight,
        z >= var - portfolio_returns,  # z >= var - portfolio_return (losses beyond VaR)
        z >= 0,  # z >= 0 (already enforced by nonneg=True, but explicit is clearer)
        var >= var_lower,  # Bound var to prevent unboundedness
        var <= var_upper
    ]
    
    if target_return is not None:
        mean_returns = np.mean(returns, axis=0)
        constraints.append(mean_returns @ w >= target_return)
    
    # Solve - try multiple solvers
    problem = cp.Problem(objective, constraints)
    
    # Try solvers in order of preference (check if they exist first)
    solvers_to_try = []
    if hasattr(cp, 'CLARABEL'):
        solvers_to_try.append(cp.CLARABEL)
    if hasattr(cp, 'ECOS'):
        solvers_to_try.append(cp.ECOS)
    if hasattr(cp, 'OSQP'):
        solvers_to_try.append(cp.OSQP)
    if hasattr(cp, 'SCS'):
        solvers_to_try.append(cp.SCS)
    
    solved = False
    
    for solver in solvers_to_try:
        try:
            problem.solve(solver=solver)
            if problem.status in ["optimal", "optimal_inaccurate"]:
                solved = True
                break
        except Exception as e:
            continue
    
    if not solved:
        # Try without specifying solver (cvxpy will choose automatically)
        try:
            problem.solve()
        except Exception as e:
            raise ValueError(f"All solvers failed. Last error: {e}")
    
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        # If optimization failed, try a simpler approach: minimize empirical CVaR directly
        # This is a fallback method
        print(f"Warning: CVaR LP optimization failed with status: {problem.status}")
        print("Falling back to empirical CVaR minimization...")
        
        # Fallback: Use scipy optimization to minimize empirical CVaR
        from scipy.optimize import minimize
        
        def empirical_cvar_objective(weights):
            weights = np.array(weights)
            portfolio_returns = returns @ weights
            var_threshold = -np.percentile(portfolio_returns, alpha * 100)
            tail_losses = portfolio_returns[portfolio_returns <= -var_threshold]
            if len(tail_losses) > 0:
                cvar = -np.mean(tail_losses)
            else:
                cvar = var_threshold
            return cvar
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        bounds = [(min_weight, max_weight) for _ in range(n_assets)]
        
        # Initial guess: equal weights
        x0 = np.ones(n_assets) / n_assets
        
        result = minimize(empirical_cvar_objective, x0, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if not result.success:
            raise ValueError(f"Fallback optimization also failed: {result.message}")
        
        optimal_weights = result.x
        optimal_var = None  # Not available from fallback
    else:
        optimal_weights = w.value
        optimal_var = var.value
    
    # Calculate actual CVaR
    portfolio_returns_actual = returns @ optimal_weights
    var_threshold = -np.percentile(portfolio_returns_actual, alpha * 100)
    tail_losses = portfolio_returns_actual[portfolio_returns_actual <= -var_threshold]
    actual_cvar = -np.mean(tail_losses) if len(tail_losses) > 0 else var_threshold
    
    # Use calculated var_threshold if optimal_var is None (from fallback)
    if optimal_var is None:
        optimal_var = var_threshold
    
    # Calculate other metrics
    mean_return = np.mean(portfolio_returns_actual)
    std_return = np.std(portfolio_returns_actual)
    sharpe_ratio = mean_return / std_return if std_return > 0 else 0
    
    return {
        "weights": optimal_weights,
        "cvar": actual_cvar,
        "var": var_threshold,
        "expected_return": mean_return,
        "volatility": std_return,
        "sharpe_ratio": sharpe_ratio,
        "confidence_level": confidence_level,
        "status": problem.status
    }


if __name__ == "__main__":
    # Test
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from data.preprocessing_pipeline import PreprocessingPipeline
    
    pipeline = PreprocessingPipeline()
    symbols = ["RELIANCE", "TCS", "HDFCBANK"]
    data = pipeline.load_and_preprocess(symbols)
    
    returns = data["returns_df"].values
    
    result = cvar_optimization(returns, confidence_level=0.95)
    
    print("CVaR Optimization Results:")
    print(f"  CVaR: {result['cvar']:.4f}")
    print(f"  VaR: {result['var']:.4f}")
    print(f"  Expected Return: {result['expected_return']:.4f}")
    print(f"  Volatility: {result['volatility']:.4f}")
    print(f"  Sharpe Ratio: {result['sharpe_ratio']:.4f}")
    print(f"  Weights: {result['weights']}")

