"""
Variance-Covariance (Parametric) VaR calculation module.

Implements rolling VaR calculation using the parametric (variance-covariance) method.
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Union, Optional, List, Tuple


def compute_var(
    returns: Union[pd.Series, pd.DataFrame, np.ndarray],
    portfolio_weights: Optional[np.ndarray] = None,
    confidence_level: float = 0.95,
    horizon: int = 1,
    annualize: bool = True
) -> Union[float, pd.Series]:
    """
    Calculate Parametric (Variance-Covariance) Value-at-Risk.
    
    Assumes returns follow a normal distribution.
    VaR = -μ - z_α * σ * sqrt(horizon)
    
    Args:
        returns: Returns data (can be Series, DataFrame, or array)
        portfolio_weights: Portfolio weights (if None, calculates per-asset VaR)
        confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
        horizon: Time horizon in days
        annualize: Whether returns are annualized
        
    Returns:
        VaR value(s)
    """
    if isinstance(returns, pd.DataFrame):
        returns_array = returns.values
    elif isinstance(returns, pd.Series):
        returns_array = returns.values.reshape(-1, 1)
    else:
        returns_array = np.array(returns)
        if returns_array.ndim == 1:
            returns_array = returns_array.reshape(-1, 1)
    
    # Calculate z-score for confidence level
    alpha = 1 - confidence_level
    z_score = stats.norm.ppf(confidence_level)
    
    # Calculate mean and std
    mean_returns = np.mean(returns_array, axis=0)
    std_returns = np.std(returns_array, axis=0, ddof=1)
    
    # Annualize if needed
    if annualize:
        mean_returns = mean_returns * 252
        std_returns = std_returns * np.sqrt(252)
    
    # Calculate VaR per asset
    var_per_asset = -mean_returns - z_score * std_returns * np.sqrt(horizon)
    
    # If portfolio weights provided, calculate portfolio VaR
    if portfolio_weights is not None:
        portfolio_weights = np.array(portfolio_weights)
        if len(portfolio_weights) != returns_array.shape[1]:
            raise ValueError("Portfolio weights length must match number of assets")
        
        # Portfolio mean return
        portfolio_mean = np.dot(portfolio_weights, mean_returns)
        
        # Portfolio variance (need covariance matrix)
        if isinstance(returns, pd.DataFrame):
            cov_matrix = returns.cov().values
            if annualize:
                cov_matrix = cov_matrix * 252
        else:
            cov_matrix = np.cov(returns_array.T)
            if annualize:
                cov_matrix = cov_matrix * 252
        
        portfolio_variance = np.dot(portfolio_weights, np.dot(cov_matrix, portfolio_weights))
        portfolio_std = np.sqrt(portfolio_variance)
        
        portfolio_var = -portfolio_mean - z_score * portfolio_std * np.sqrt(horizon)
        return portfolio_var
    
    # Return per-asset VaR
    if isinstance(returns, pd.Series):
        return var_per_asset[0]
    elif isinstance(returns, pd.DataFrame):
        return pd.Series(var_per_asset, index=returns.columns)
    else:
        return var_per_asset


def compute_rolling_var(
    returns: pd.DataFrame,
    portfolio_weights: pd.Series,
    window: int = 252,
    confidence_level: float = 0.95,
    horizon: int = 1,
    min_periods: Optional[int] = None
) -> pd.Series:
    """
    Compute rolling VaR using a rolling window.
    
    Args:
        returns: DataFrame of asset returns with dates as index
        portfolio_weights: Series of portfolio weights with assets as index
        window: Rolling window size in days
        confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
        horizon: Time horizon in days
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
    
    # Compute rolling portfolio returns
    portfolio_returns = (returns_aligned * weights_aligned).sum(axis=1)
    
    # Compute rolling VaR
    rolling_var = pd.Series(index=returns.index, dtype=float)
    
    for i in range(len(returns)):
        if i < min_periods - 1:
            rolling_var.iloc[i] = np.nan
            continue
        
        # Get window of returns
        start_idx = max(0, i - window + 1)
        window_returns = portfolio_returns.iloc[start_idx:i+1]
        
        if len(window_returns) < min_periods:
            rolling_var.iloc[i] = np.nan
            continue
        
        # Calculate VaR for this window
        var_value = compute_var(
            window_returns,
            confidence_level=confidence_level,
            horizon=horizon,
            annualize=False  # Already daily returns
        )
        
        rolling_var.iloc[i] = var_value
    
    return rolling_var


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

