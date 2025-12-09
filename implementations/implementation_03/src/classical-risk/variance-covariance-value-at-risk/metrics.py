"""
Metrics computation module for VaR evaluation.

Computes tail risk, portfolio structure, distribution, and runtime metrics.
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Optional
import time


def compute_tail_metrics(
    returns: pd.Series,
    var_series: pd.Series
) -> Dict[str, float]:
    """
    Compute tail risk metrics including exceedances.
    
    Args:
        returns: Series of actual returns
        var_series: Series of VaR values
        
    Returns:
        Dictionary of tail metrics
    """
    violations = returns < -var_series
    
    if violations.sum() == 0:
        return {
            'mean_exceedance': np.nan,
            'max_exceedance': np.nan,
            'std_exceedance': np.nan,
            'quantile_loss_score': np.nan,
            'rmse_var_vs_losses': np.nan
        }
    
    # Exceedances (actual losses beyond VaR)
    exceedances = (-returns - var_series)[violations]
    
    mean_exceedance = exceedances.mean()
    max_exceedance = exceedances.max()
    std_exceedance = exceedances.std()
    
    # Quantile loss (pinball loss) for VaR
    alpha = 1 - 0.95  # Assuming 95% VaR, adjust if needed
    quantile_loss = np.mean(
        np.maximum(alpha * (returns + var_series), (alpha - 1) * (returns + var_series))
    )
    
    # RMSE between VaR and actual losses (only for violations)
    rmse_var_vs_losses = np.sqrt(np.mean((var_series[violations] + returns[violations])**2))
    
    return {
        'mean_exceedance': mean_exceedance,
        'max_exceedance': max_exceedance,
        'std_exceedance': std_exceedance,
        'quantile_loss_score': quantile_loss,
        'rmse_var_vs_losses': rmse_var_vs_losses
    }


def compute_structure_metrics(
    portfolio_weights: pd.Series,
    covariance_matrix: Optional[pd.DataFrame] = None
) -> Dict[str, float]:
    """
    Compute portfolio structure metrics.
    
    Args:
        portfolio_weights: Series of portfolio weights
        covariance_matrix: Optional covariance matrix for condition number
        
    Returns:
        Dictionary of structure metrics
    """
    weights_array = portfolio_weights.values
    active_weights = weights_array[weights_array > 1e-10]  # Non-zero weights
    
    num_active_assets = len(active_weights)
    
    # HHI (Herfindahl-Hirschman Index) - concentration measure
    hhi = np.sum(weights_array**2)
    
    # Effective number of assets (inverse of HHI)
    enc = 1 / hhi if hhi > 0 else np.nan
    
    # Condition number of covariance matrix
    condition_number = np.nan
    if covariance_matrix is not None:
        try:
            cov_array = covariance_matrix.values
            eigenvals = np.linalg.eigvals(cov_array)
            eigenvals = eigenvals[eigenvals > 1e-10]  # Remove near-zero eigenvalues
            if len(eigenvals) > 0:
                condition_number = np.max(eigenvals) / np.min(eigenvals)
        except:
            condition_number = np.nan
    
    return {
        'portfolio_size': num_active_assets,  # Number of assets with non-zero weights
        'num_active_assets': num_active_assets,  # Alias for portfolio_size
        'hhi_concentration': hhi,
        'effective_number_of_assets': enc,
        'covariance_condition_number': condition_number
    }


def compute_distribution_metrics(returns: pd.Series) -> Dict[str, float]:
    """
    Compute distribution metrics for returns.
    
    Args:
        returns: Series of returns
        
    Returns:
        Dictionary of distribution metrics
    """
    returns_clean = returns.dropna()
    
    if len(returns_clean) < 3:
        return {
            'skewness': np.nan,
            'kurtosis': np.nan,
            'jarque_bera_p_value': np.nan
        }
    
    skewness = stats.skew(returns_clean)
    kurtosis = stats.kurtosis(returns_clean)  # Excess kurtosis
    
    # Jarque-Bera test for normality
    jb_stat, jb_p_value = stats.jarque_bera(returns_clean)
    
    return {
        'skewness': skewness,
        'kurtosis': kurtosis,
        'jarque_bera_p_value': jb_p_value,
        'jarque_bera_statistic': jb_stat
    }


def compute_runtime_metrics(runtimes: list[float]) -> Dict[str, float]:
    """
    Compute runtime performance metrics.
    
    Args:
        runtimes: List of runtime values in seconds
        
    Returns:
        Dictionary of runtime metrics
    """
    if len(runtimes) == 0:
        return {
            'runtime_per_portfolio_ms': np.nan,
            'p95_runtime_ms': np.nan,
            'mean_runtime_ms': np.nan,
            'median_runtime_ms': np.nan
        }
    
    runtimes_array = np.array(runtimes)
    
    return {
        'runtime_per_portfolio_ms': np.mean(runtimes_array) * 1000,
        'p95_runtime_ms': np.percentile(runtimes_array, 95) * 1000,
        'mean_runtime_ms': np.mean(runtimes_array) * 1000,
        'median_runtime_ms': np.median(runtimes_array) * 1000,
        'min_runtime_ms': np.min(runtimes_array) * 1000,
        'max_runtime_ms': np.max(runtimes_array) * 1000
    }

