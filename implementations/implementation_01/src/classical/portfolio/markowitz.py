"""
Markowitz Mean-Variance Portfolio Optimization.
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
import cvxpy as cp


def markowitz_optimization(
    mean_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    risk_aversion: float = 1.0,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    target_return: Optional[float] = None,
    target_volatility: Optional[float] = None
) -> Dict:
    """
    Solve Markowitz mean-variance optimization problem.
    
    Maximize: μ^T w - λ * w^T Σ w
    Subject to: sum(w) = 1, min_weight <= w <= max_weight
    
    Args:
        mean_returns: Expected returns vector (N,)
        covariance_matrix: Covariance matrix (N x N)
        risk_aversion: Risk aversion parameter (lambda)
        min_weight: Minimum weight per asset
        max_weight: Maximum weight per asset
        target_return: Optional target return constraint
        target_volatility: Optional target volatility constraint
        
    Returns:
        Dictionary with optimal weights, expected return, volatility, Sharpe ratio
    """
    n_assets = len(mean_returns)
    
    # Decision variables
    w = cp.Variable(n_assets)
    
    # Objective: maximize return - risk_aversion * risk
    portfolio_return = mean_returns @ w
    portfolio_variance = cp.quad_form(w, covariance_matrix)
    portfolio_std = cp.sqrt(portfolio_variance)
    
    objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_variance)
    
    # Constraints
    constraints = [
        cp.sum(w) == 1,  # Budget constraint
        w >= min_weight,  # Long-only (or minimum weight)
        w <= max_weight   # Maximum weight
    ]
    
    # Optional constraints
    if target_return is not None:
        constraints.append(portfolio_return >= target_return)
    
    if target_volatility is not None:
        constraints.append(portfolio_std <= target_volatility)
    
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
        raise ValueError(f"Optimization failed with status: {problem.status}")
    
    optimal_weights = w.value
    
    # Calculate metrics
    expected_return = np.dot(mean_returns, optimal_weights)
    portfolio_variance_val = np.dot(optimal_weights, np.dot(covariance_matrix, optimal_weights))
    portfolio_std_val = np.sqrt(portfolio_variance_val)
    sharpe_ratio = expected_return / portfolio_std_val if portfolio_std_val > 0 else 0
    
    return {
        "weights": optimal_weights,
        "expected_return": expected_return,
        "volatility": portfolio_std_val,
        "variance": portfolio_variance_val,
        "sharpe_ratio": sharpe_ratio,
        "risk_aversion": risk_aversion,
        "status": problem.status
    }


def efficient_frontier(
    mean_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    n_points: int = 50,
    min_weight: float = 0.0,
    max_weight: float = 1.0
) -> pd.DataFrame:
    """
    Generate efficient frontier.
    
    Args:
        mean_returns: Expected returns
        covariance_matrix: Covariance matrix
        n_points: Number of points on frontier
        min_weight: Minimum weight
        max_weight: Maximum weight
        
    Returns:
        DataFrame with return, volatility, and weights for each point
    """
    # Find min and max returns
    n_assets = len(mean_returns)
    
    # Minimum variance portfolio
    w_min_var = cp.Variable(n_assets)
    prob_min_var = cp.Problem(
        cp.Minimize(cp.quad_form(w_min_var, covariance_matrix)),
        [cp.sum(w_min_var) == 1, w_min_var >= min_weight, w_min_var <= max_weight]
    )
    # Try multiple solvers
    solvers_to_try = []
    if hasattr(cp, 'CLARABEL'):
        solvers_to_try.append(cp.CLARABEL)
    if hasattr(cp, 'ECOS'):
        solvers_to_try.append(cp.ECOS)
    if hasattr(cp, 'OSQP'):
        solvers_to_try.append(cp.OSQP)
    if hasattr(cp, 'SCS'):
        solvers_to_try.append(cp.SCS)
    
    for solver in solvers_to_try:
        try:
            prob_min_var.solve(solver=solver)
            if prob_min_var.status in ["optimal", "optimal_inaccurate"]:
                break
        except:
            continue
    if prob_min_var.status not in ["optimal", "optimal_inaccurate"]:
        prob_min_var.solve()  # Let cvxpy choose
    min_var_return = mean_returns @ w_min_var.value
    
    # Maximum return portfolio
    w_max_ret = cp.Variable(n_assets)
    prob_max_ret = cp.Problem(
        cp.Maximize(mean_returns @ w_max_ret),
        [cp.sum(w_max_ret) == 1, w_max_ret >= min_weight, w_max_ret <= max_weight]
    )
    # Try multiple solvers
    solvers_to_try = []
    if hasattr(cp, 'CLARABEL'):
        solvers_to_try.append(cp.CLARABEL)
    if hasattr(cp, 'ECOS'):
        solvers_to_try.append(cp.ECOS)
    if hasattr(cp, 'OSQP'):
        solvers_to_try.append(cp.OSQP)
    if hasattr(cp, 'SCS'):
        solvers_to_try.append(cp.SCS)
    
    for solver in solvers_to_try:
        try:
            prob_max_ret.solve(solver=solver)
            if prob_max_ret.status in ["optimal", "optimal_inaccurate"]:
                break
        except:
            continue
    if prob_max_ret.status not in ["optimal", "optimal_inaccurate"]:
        prob_max_ret.solve()  # Let cvxpy choose
    max_ret_return = mean_returns @ w_max_ret.value
    
    # Generate target returns
    target_returns = np.linspace(min_var_return, max_ret_return, n_points)
    
    # Solve for each target return
    results = []
    for target_ret in target_returns:
        try:
            result = markowitz_optimization(
                mean_returns, covariance_matrix,
                risk_aversion=1.0,
                min_weight=min_weight,
                max_weight=max_weight,
                target_return=target_ret
            )
            results.append({
                "return": result["expected_return"],
                "volatility": result["volatility"],
                "sharpe_ratio": result["sharpe_ratio"],
                **{f"weight_{i}": result["weights"][i] for i in range(n_assets)}
            })
        except:
            continue
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Test
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from data.preprocessing_pipeline import PreprocessingPipeline
    
    pipeline = PreprocessingPipeline()
    symbols = ["RELIANCE", "TCS", "HDFCBANK", "HDFCBANK", "ICICIBANK"]
    data = pipeline.load_and_preprocess(symbols)
    
    mu = data["mean_returns"].values
    Sigma = data["covariance_matrix"].values
    
    # Optimize
    result = markowitz_optimization(mu, Sigma, risk_aversion=2.0)
    
    print("Markowitz Optimization Results:")
    print(f"  Expected Return: {result['expected_return']:.4f}")
    print(f"  Volatility: {result['volatility']:.4f}")
    print(f"  Sharpe Ratio: {result['sharpe_ratio']:.4f}")
    print(f"  Weights: {result['weights']}")

