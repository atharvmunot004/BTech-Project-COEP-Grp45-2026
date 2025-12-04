"""
Monte Carlo Simulation for VaR and CVaR estimation.
"""
import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple
from scipy import stats


def monte_carlo_var_cvar(
    returns: Union[pd.Series, pd.DataFrame, np.ndarray],
    portfolio_weights: Optional[np.ndarray] = None,
    confidence_level: float = 0.95,
    horizon: int = 1,
    n_simulations: int = 10000,
    random_seed: Optional[int] = None
) -> Tuple[float, float, np.ndarray]:
    """
    Calculate VaR and CVaR using Monte Carlo simulation.
    
    Args:
        returns: Historical returns data
        portfolio_weights: Portfolio weights (if None, uses equal weights)
        confidence_level: Confidence level (e.g., 0.95)
        horizon: Time horizon in days
        n_simulations: Number of Monte Carlo simulations
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (VaR, CVaR, simulated_returns)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Convert to numpy array
    if isinstance(returns, pd.DataFrame):
        returns_array = returns.values
        symbols = returns.columns.tolist()
    elif isinstance(returns, pd.Series):
        returns_array = returns.values.reshape(-1, 1)
        symbols = [returns.name] if returns.name else [0]
    else:
        returns_array = np.array(returns)
        if returns_array.ndim == 1:
            returns_array = returns_array.reshape(-1, 1)
        symbols = list(range(returns_array.shape[1]))
    
    n_assets = returns_array.shape[1]
    n_periods = returns_array.shape[0]
    
    # Calculate mean and covariance
    mean_returns = np.mean(returns_array, axis=0)
    if isinstance(returns, pd.DataFrame):
        cov_matrix = returns.cov().values
    else:
        cov_matrix = np.cov(returns_array.T)
    
    # Annualize
    mean_returns = mean_returns * 252
    cov_matrix = cov_matrix * 252
    
    # Portfolio weights
    if portfolio_weights is None:
        portfolio_weights = np.ones(n_assets) / n_assets
    else:
        portfolio_weights = np.array(portfolio_weights)
        if len(portfolio_weights) != n_assets:
            raise ValueError("Portfolio weights length must match number of assets")
    
    # Simulate returns
    simulated_returns = np.random.multivariate_normal(
        mean_returns,
        cov_matrix,
        size=(n_simulations, horizon)
    )
    
    # Calculate portfolio returns for each simulation
    if horizon == 1:
        portfolio_sim_returns = np.dot(simulated_returns, portfolio_weights)
    else:
        # For multi-period, compound returns
        portfolio_sim_returns = np.zeros(n_simulations)
        for i in range(n_simulations):
            portfolio_return = 0
            for t in range(horizon):
                period_return = np.dot(simulated_returns[i, t], portfolio_weights)
                portfolio_return += period_return
            portfolio_sim_returns[i] = portfolio_return
    
    # Calculate VaR (percentile)
    alpha = 1 - confidence_level
    var = -np.percentile(portfolio_sim_returns, alpha * 100)
    
    # Calculate CVaR (expected shortfall)
    threshold = -var
    tail_losses = portfolio_sim_returns[portfolio_sim_returns <= -var]
    if len(tail_losses) > 0:
        cvar = -np.mean(tail_losses)
    else:
        cvar = var  # Fallback if no tail losses
    
    return var, cvar, portfolio_sim_returns


def historical_var_cvar(
    returns: Union[pd.Series, pd.DataFrame, np.ndarray],
    portfolio_weights: Optional[np.ndarray] = None,
    confidence_level: float = 0.95,
    window: Optional[int] = None
) -> Tuple[float, float]:
    """
    Calculate Historical VaR and CVaR (non-parametric).
    
    Args:
        returns: Historical returns
        portfolio_weights: Portfolio weights
        confidence_level: Confidence level
        window: Rolling window (if None, uses all data)
        
    Returns:
        Tuple of (VaR, CVaR)
    """
    # Convert to numpy array
    if isinstance(returns, pd.DataFrame):
        returns_array = returns.values
    elif isinstance(returns, pd.Series):
        returns_array = returns.values.reshape(-1, 1)
    else:
        returns_array = np.array(returns)
        if returns_array.ndim == 1:
            returns_array = returns_array.reshape(-1, 1)
    
    # Apply window if specified
    if window is not None:
        returns_array = returns_array[-window:]
    
    # Portfolio weights
    if portfolio_weights is None:
        portfolio_weights = np.ones(returns_array.shape[1]) / returns_array.shape[1]
    else:
        portfolio_weights = np.array(portfolio_weights)
    
    # Calculate portfolio returns
    portfolio_returns = np.dot(returns_array, portfolio_weights)
    
    # Calculate VaR
    alpha = 1 - confidence_level
    var = -np.percentile(portfolio_returns, alpha * 100)
    
    # Calculate CVaR
    tail_losses = portfolio_returns[portfolio_returns <= -var]
    if len(tail_losses) > 0:
        cvar = -np.mean(tail_losses)
    else:
        cvar = var
    
    return var, cvar


if __name__ == "__main__":
    # Test
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from data.preprocessing_pipeline import PreprocessingPipeline
    
    pipeline = PreprocessingPipeline()
    symbols = ["RELIANCE", "TCS", "HDFCBANK"]
    data = pipeline.load_and_preprocess(symbols)
    
    returns = data["returns_df"]
    weights = np.array([1/3, 1/3, 1/3])
    
    # Monte Carlo VaR/CVaR
    var_mc, cvar_mc, sim_returns = monte_carlo_var_cvar(
        returns, portfolio_weights=weights, n_simulations=10000, random_seed=42
    )
    print(f"Monte Carlo VaR (95%): {var_mc:.4f}")
    print(f"Monte Carlo CVaR (95%): {cvar_mc:.4f}")
    
    # Historical VaR/CVaR
    var_hist, cvar_hist = historical_var_cvar(returns, portfolio_weights=weights)
    print(f"\nHistorical VaR (95%): {var_hist:.4f}")
    print(f"Historical CVaR (95%): {cvar_hist:.4f}")

