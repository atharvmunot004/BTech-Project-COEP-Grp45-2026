"""
Monte Carlo Simulation for VaR and CVaR calculation module.

Implements rolling VaR and CVaR calculation using Monte Carlo simulation.
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Union, Optional, List, Tuple
import warnings


def simulate_portfolio_returns(
    returns_window: pd.DataFrame,
    portfolio_weights: np.ndarray,
    num_simulations: int = 10000,
    horizon: int = 1,
    distribution_type: str = 'multivariate_normal',
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Simulate portfolio returns using Monte Carlo method.
    
    Args:
        returns_window: DataFrame of historical returns (window size x num assets)
        portfolio_weights: Array of portfolio weights
        num_simulations: Number of Monte Carlo simulations
        horizon: Time horizon in days
        distribution_type: Distribution assumption ('multivariate_normal')
        random_seed: Random seed for reproducibility
        
    Returns:
        Array of simulated portfolio returns (num_simulations,)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Align assets
    returns_array = returns_window.values  # (window_size, num_assets)
    
    if distribution_type == 'multivariate_normal':
        # Estimate mean and covariance from historical returns
        mean_returns = returns_array.mean(axis=0)
        cov_matrix = np.cov(returns_array.T)
        
        # Ensure covariance matrix is positive semi-definite
        cov_matrix = (cov_matrix + cov_matrix.T) / 2
        
        # Add small regularization to avoid singular matrix issues
        cov_matrix = cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-8
        
        # Check for singular matrix
        try:
            # Simulate asset returns
            simulated_asset_returns = np.random.multivariate_normal(
                mean_returns,
                cov_matrix,
                size=(num_simulations, horizon)
            )
        except np.linalg.LinAlgError:
            # If matrix is still singular, use diagonal approximation
            cov_matrix = np.diag(np.diag(cov_matrix))
            simulated_asset_returns = np.random.multivariate_normal(
                mean_returns,
                cov_matrix,
                size=(num_simulations, horizon)
            )
        
        # If horizon > 1, sum returns over horizon
        if horizon > 1:
            simulated_asset_returns = simulated_asset_returns.sum(axis=1)
        else:
            simulated_asset_returns = simulated_asset_returns.squeeze()
        
        # Compute portfolio returns: weighted sum
        portfolio_returns = np.dot(simulated_asset_returns, portfolio_weights)
        
    else:
        raise ValueError(f"Unsupported distribution type: {distribution_type}")
    
    return portfolio_returns


def compute_var_cvar_from_simulations(
    simulated_returns: np.ndarray,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Compute VaR and CVaR from simulated returns.
    
    Args:
        simulated_returns: Array of simulated portfolio returns
        confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
        
    Returns:
        Tuple of (VaR, CVaR)
    """
    # Sort returns (ascending order - losses are negative)
    sorted_returns = np.sort(simulated_returns)
    
    # Calculate VaR (Value at Risk) - quantile of losses
    alpha = 1 - confidence_level
    var_index = int(np.floor(alpha * len(sorted_returns)))
    var_index = max(0, min(var_index, len(sorted_returns) - 1))
    
    var = -sorted_returns[var_index]  # VaR is positive (loss amount)
    
    # Calculate CVaR (Conditional Value at Risk) - expected loss beyond VaR
    # CVaR is the mean of returns in the tail (worst alpha% of outcomes)
    tail_returns = sorted_returns[:var_index + 1]
    cvar = -tail_returns.mean() if len(tail_returns) > 0 else var
    
    return var, cvar


def compute_rolling_var(
    returns: pd.DataFrame,
    portfolio_weights: pd.Series,
    window: int = 252,
    confidence_level: float = 0.95,
    horizon: int = 1,
    num_simulations: int = 10000,
    distribution_type: str = 'multivariate_normal',
    random_seed: Optional[int] = None,
    min_periods: Optional[int] = None
) -> pd.Series:
    """
    Compute rolling VaR using Monte Carlo simulation.
    
    Args:
        returns: DataFrame of asset returns with dates as index
        portfolio_weights: Series of portfolio weights with assets as index
        window: Rolling window size in days
        confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
        horizon: Time horizon in days
        num_simulations: Number of Monte Carlo simulations
        distribution_type: Distribution assumption
        random_seed: Random seed for reproducibility
        min_periods: Minimum number of periods required for calculation
        
    Returns:
        Series of rolling VaR values with dates as index
    """
    if min_periods is None:
        min_periods = min(window, len(returns))
    
    # Align assets
    common_assets = returns.columns.intersection(portfolio_weights.index)
    if len(common_assets) == 0:
        raise ValueError("No common assets between returns and weights")
    
    returns_aligned = returns[common_assets]
    weights_aligned = portfolio_weights[common_assets].values
    weights_aligned = weights_aligned / weights_aligned.sum()  # Normalize
    
    # Compute rolling VaR
    rolling_var = pd.Series(index=returns.index, dtype=float)
    
    for i in range(len(returns_aligned)):
        if i < min_periods - 1:
            rolling_var.iloc[i] = np.nan
            continue
        
        # Get window of returns
        start_idx = max(0, i - window + 1)
        window_returns = returns_aligned.iloc[start_idx:i+1]
        
        if len(window_returns) < min_periods:
            rolling_var.iloc[i] = np.nan
            continue
        
        try:
            # Simulate portfolio returns
            simulated_returns = simulate_portfolio_returns(
                window_returns,
                weights_aligned,
                num_simulations=num_simulations,
                horizon=horizon,
                distribution_type=distribution_type,
                random_seed=random_seed
            )
            
            # Compute VaR from simulations
            var, _ = compute_var_cvar_from_simulations(
                simulated_returns,
                confidence_level=confidence_level
            )
            
            rolling_var.iloc[i] = var
            
        except Exception as e:
            rolling_var.iloc[i] = np.nan
    
    return rolling_var


def compute_rolling_cvar(
    returns: pd.DataFrame,
    portfolio_weights: pd.Series,
    window: int = 252,
    confidence_level: float = 0.95,
    horizon: int = 1,
    num_simulations: int = 10000,
    distribution_type: str = 'multivariate_normal',
    random_seed: Optional[int] = None,
    min_periods: Optional[int] = None
) -> pd.Series:
    """
    Compute rolling CVaR using Monte Carlo simulation.
    
    Args:
        returns: DataFrame of asset returns with dates as index
        portfolio_weights: Series of portfolio weights with assets as index
        window: Rolling window size in days
        confidence_level: Confidence level (e.g., 0.95 for 95% CVaR)
        horizon: Time horizon in days
        num_simulations: Number of Monte Carlo simulations
        distribution_type: Distribution assumption
        random_seed: Random seed for reproducibility
        min_periods: Minimum number of periods required for calculation
        
    Returns:
        Series of rolling CVaR values with dates as index
    """
    if min_periods is None:
        min_periods = min(window, len(returns))
    
    # Align assets
    common_assets = returns.columns.intersection(portfolio_weights.index)
    if len(common_assets) == 0:
        raise ValueError("No common assets between returns and weights")
    
    returns_aligned = returns[common_assets]
    weights_aligned = portfolio_weights[common_assets].values
    weights_aligned = weights_aligned / weights_aligned.sum()  # Normalize
    
    # Compute rolling CVaR
    rolling_cvar = pd.Series(index=returns.index, dtype=float)
    
    for i in range(len(returns_aligned)):
        if i < min_periods - 1:
            rolling_cvar.iloc[i] = np.nan
            continue
        
        # Get window of returns
        start_idx = max(0, i - window + 1)
        window_returns = returns_aligned.iloc[start_idx:i+1]
        
        if len(window_returns) < min_periods:
            rolling_cvar.iloc[i] = np.nan
            continue
        
        try:
            # Simulate portfolio returns
            simulated_returns = simulate_portfolio_returns(
                window_returns,
                weights_aligned,
                num_simulations=num_simulations,
                horizon=horizon,
                distribution_type=distribution_type,
                random_seed=random_seed
            )
            
            # Compute CVaR from simulations
            _, cvar = compute_var_cvar_from_simulations(
                simulated_returns,
                confidence_level=confidence_level
            )
            
            rolling_cvar.iloc[i] = cvar
            
        except Exception as e:
            rolling_cvar.iloc[i] = np.nan
    
    return rolling_cvar


def align_returns_and_var(
    returns: pd.Series,
    var_series: pd.Series
) -> Tuple[pd.Series, pd.Series]:
    """
    Align returns and VaR series to common dates.
    
    Args:
        returns: Series of returns with dates as index
        var_series: Series of VaR values with dates as index
        
    Returns:
        Tuple of (aligned_returns, aligned_var)
    """
    # Find common dates
    common_dates = returns.index.intersection(var_series.index)
    
    if len(common_dates) == 0:
        raise ValueError("No common dates between returns and VaR series")
    
    aligned_returns = returns.loc[common_dates]
    aligned_var = var_series.loc[common_dates]
    
    return aligned_returns, aligned_var


def align_returns_and_cvar(
    returns: pd.Series,
    cvar_series: pd.Series
) -> Tuple[pd.Series, pd.Series]:
    """
    Align returns and CVaR series to common dates.
    
    Args:
        returns: Series of returns with dates as index
        cvar_series: Series of CVaR values with dates as index
        
    Returns:
        Tuple of (aligned_returns, aligned_cvar)
    """
    # Find common dates
    common_dates = returns.index.intersection(cvar_series.index)
    
    if len(common_dates) == 0:
        raise ValueError("No common dates between returns and CVaR series")
    
    aligned_returns = returns.loc[common_dates]
    aligned_cvar = cvar_series.loc[common_dates]
    
    return aligned_returns, aligned_cvar

