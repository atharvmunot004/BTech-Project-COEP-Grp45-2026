"""
Metrics computation module for Markowitz optimization.

Computes portfolio quality, structure, risk, distribution, and runtime metrics.
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Optional, Tuple
import time


def compute_portfolio_statistics(
    portfolio_returns: pd.Series,
    portfolio_weights: pd.Series,
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0
) -> Dict[str, float]:
    """
    Compute portfolio quality statistics.
    
    Args:
        portfolio_returns: Series of portfolio returns
        portfolio_weights: Series of portfolio weights
        expected_returns: Series of expected returns
        covariance_matrix: Covariance matrix DataFrame
        risk_free_rate: Risk-free rate
        
    Returns:
        Dictionary of portfolio statistics
    """
    # Align assets
    assets = portfolio_weights.index.intersection(expected_returns.index)
    assets = assets.intersection(covariance_matrix.index)
    
    if len(assets) == 0:
        return {
            'expected_return': np.nan,
            'volatility': np.nan,
            'portfolio_variance': np.nan,
            'sharpe_ratio': np.nan
        }
    
    weights = portfolio_weights[assets].values
    mu = expected_returns[assets].values
    Sigma = covariance_matrix.loc[assets, assets].values
    
    # Expected return
    expected_return = mu @ weights
    
    # Portfolio variance
    portfolio_variance = weights @ Sigma @ weights
    
    # Volatility (standard deviation)
    volatility = np.sqrt(portfolio_variance)
    
    # Sharpe ratio
    sharpe_ratio = (expected_return - risk_free_rate) / volatility if volatility > 0 else np.nan
    
    return {
        'expected_return': expected_return,
        'volatility': volatility,
        'portfolio_variance': portfolio_variance,
        'sharpe_ratio': sharpe_ratio
    }


def compute_sharpe_ratio(
    portfolio_returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualize: bool = True
) -> float:
    """
    Compute Sharpe ratio from portfolio returns.
    
    Args:
        portfolio_returns: Series of portfolio returns
        risk_free_rate: Risk-free rate
        annualize: Whether to annualize the ratio
        
    Returns:
        Sharpe ratio
    """
    returns_clean = portfolio_returns.dropna()
    
    if len(returns_clean) < 2:
        return np.nan
    
    mean_return = returns_clean.mean()
    std_return = returns_clean.std()
    
    if annualize:
        mean_return = mean_return * 252
        std_return = std_return * np.sqrt(252)
    
    if std_return == 0:
        return np.nan
    
    sharpe = (mean_return - risk_free_rate) / std_return
    return sharpe


def compute_sortino_ratio(
    portfolio_returns: pd.Series,
    risk_free_rate: float = 0.0,
    target_return: float = 0.0,
    annualize: bool = True
) -> float:
    """
    Compute Sortino ratio (downside risk-adjusted return).
    
    Args:
        portfolio_returns: Series of portfolio returns
        risk_free_rate: Risk-free rate
        target_return: Target return for downside deviation
        annualize: Whether to annualize the ratio
        
    Returns:
        Sortino ratio
    """
    returns_clean = portfolio_returns.dropna()
    
    if len(returns_clean) < 2:
        return np.nan
    
    mean_return = returns_clean.mean()
    
    # Downside deviation (only negative deviations from target)
    downside_returns = returns_clean[returns_clean < target_return] - target_return
    downside_deviation = np.sqrt(np.mean(downside_returns**2)) if len(downside_returns) > 0 else 0.0
    
    if annualize:
        mean_return = mean_return * 252
        downside_deviation = downside_deviation * np.sqrt(252)
    
    if downside_deviation == 0:
        return np.nan
    
    sortino = (mean_return - risk_free_rate) / downside_deviation
    return sortino


def compute_max_drawdown(portfolio_returns: pd.Series) -> float:
    """
    Compute maximum drawdown from portfolio returns.
    
    Args:
        portfolio_returns: Series of portfolio returns
        
    Returns:
        Maximum drawdown (as positive value)
    """
    returns_clean = portfolio_returns.dropna()
    
    if len(returns_clean) == 0:
        return np.nan
    
    # Compute cumulative returns
    cumulative = (1 + returns_clean).cumprod()
    
    # Compute running maximum
    running_max = cumulative.expanding().max()
    
    # Compute drawdown
    drawdown = (cumulative - running_max) / running_max
    
    max_drawdown = abs(drawdown.min())
    return max_drawdown


def compute_calmar_ratio(
    portfolio_returns: pd.Series,
    annualize: bool = True
) -> float:
    """
    Compute Calmar ratio (annual return / max drawdown).
    
    Args:
        portfolio_returns: Series of portfolio returns
        annualize: Whether to annualize returns
        
    Returns:
        Calmar ratio
    """
    returns_clean = portfolio_returns.dropna()
    
    if len(returns_clean) == 0:
        return np.nan
    
    mean_return = returns_clean.mean()
    if annualize:
        mean_return = mean_return * 252
    
    max_dd = compute_max_drawdown(portfolio_returns)
    
    if max_dd == 0 or np.isnan(max_dd):
        return np.nan
    
    calmar = mean_return / max_dd
    return calmar


def compute_structure_metrics(
    portfolio_weights: pd.Series,
    covariance_matrix: Optional[pd.DataFrame] = None,
    returns: Optional[pd.DataFrame] = None
) -> Dict[str, float]:
    """
    Compute portfolio structure metrics.
    
    Args:
        portfolio_weights: Series of portfolio weights
        covariance_matrix: Optional covariance matrix for correlation metrics
        returns: Optional returns DataFrame for correlation metrics
        
    Returns:
        Dictionary of structure metrics
    """
    weights_array = portfolio_weights.values
    active_weights = weights_array[weights_array > 1e-10]  # Non-zero weights
    
    num_assets = len(active_weights)
    
    # HHI (Herfindahl-Hirschman Index) - concentration measure
    hhi = np.sum(weights_array**2)
    
    # Effective number of assets (inverse of HHI)
    effective_num_assets = 1 / hhi if hhi > 0 else np.nan
    
    # Weight entropy (diversification measure)
    # Higher entropy = more diversified
    weights_normalized = weights_array[weights_array > 1e-10]
    if len(weights_normalized) > 0:
        weights_normalized = weights_normalized / weights_normalized.sum()
        weight_entropy = -np.sum(weights_normalized * np.log(weights_normalized + 1e-10))
    else:
        weight_entropy = np.nan
    
    # Pairwise correlation mean
    pairwise_correlation_mean = np.nan
    if covariance_matrix is not None and returns is not None:
        try:
            # Align assets
            assets = portfolio_weights.index.intersection(covariance_matrix.index)
            assets = assets.intersection(returns.columns)
            
            if len(assets) > 1:
                # Get correlation matrix
                if hasattr(covariance_matrix, 'corr'):
                    corr_matrix = returns[assets].corr().values
                else:
                    # Compute from covariance
                    cov_array = covariance_matrix.loc[assets, assets].values
                    std_array = np.sqrt(np.diag(cov_array))
                    corr_matrix = cov_array / np.outer(std_array, std_array)
                
                # Get upper triangle (excluding diagonal)
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                pairwise_correlations = corr_matrix[mask]
                pairwise_correlation_mean = np.mean(pairwise_correlations)
        except Exception:
            pairwise_correlation_mean = np.nan
    
    return {
        'num_assets_in_portfolio': num_assets,
        'hhi_concentration': hhi,
        'effective_number_of_assets': effective_num_assets,
        'weight_entropy': weight_entropy,
        'pairwise_correlation_mean': pairwise_correlation_mean
    }


def compute_risk_metrics(
    portfolio_returns: pd.Series,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Compute risk metrics (VaR, CVaR, downside deviation).
    
    Args:
        portfolio_returns: Series of portfolio returns
        confidence_level: Confidence level for VaR/CVaR
        
    Returns:
        Dictionary of risk metrics
    """
    returns_clean = portfolio_returns.dropna()
    
    if len(returns_clean) == 0:
        return {
            'value_at_risk': np.nan,
            'conditional_value_at_risk': np.nan,
            'downside_deviation': np.nan
        }
    
    # Value at Risk (VaR) - parametric method
    mean_return = returns_clean.mean()
    std_return = returns_clean.std()
    z_score = stats.norm.ppf(confidence_level)
    var_value = -(mean_return - z_score * std_return)
    
    # Conditional Value at Risk (CVaR) - expected shortfall
    var_threshold = -var_value
    tail_returns = returns_clean[returns_clean <= var_threshold]
    if len(tail_returns) > 0:
        cvar_value = -tail_returns.mean()
    else:
        # Fallback: use parametric approximation
        alpha = 1 - confidence_level
        cvar_value = -(mean_return - (stats.norm.pdf(z_score) / alpha) * std_return)
    
    # Downside deviation (semi-deviation)
    downside_returns = returns_clean[returns_clean < 0]
    downside_deviation = np.sqrt(np.mean(downside_returns**2)) if len(downside_returns) > 0 else 0.0
    
    return {
        'value_at_risk': var_value,
        'conditional_value_at_risk': cvar_value,
        'downside_deviation': downside_deviation
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
            'jarque_bera_p_value': np.nan,
            'jarque_bera_statistic': np.nan
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


def compute_runtime_metrics(
    runtimes: list[float],
    covariance_times: Optional[list[float]] = None,
    solver_times: Optional[list[float]] = None
) -> Dict[str, float]:
    """
    Compute runtime performance metrics.
    
    Args:
        runtimes: List of total runtime values in seconds
        covariance_times: List of covariance estimation times in ms
        solver_times: List of solver times in ms
        
    Returns:
        Dictionary of runtime metrics
    """
    metrics = {}
    
    if len(runtimes) > 0:
        runtimes_array = np.array(runtimes) * 1000  # Convert to ms
        metrics['runtime_per_optimization_ms'] = np.mean(runtimes_array)
        metrics['p95_runtime_ms'] = np.percentile(runtimes_array, 95)
        metrics['median_runtime_ms'] = np.median(runtimes_array)
        metrics['min_runtime_ms'] = np.min(runtimes_array)
        metrics['max_runtime_ms'] = np.max(runtimes_array)
    else:
        metrics['runtime_per_optimization_ms'] = np.nan
        metrics['p95_runtime_ms'] = np.nan
    
    if covariance_times is not None and len(covariance_times) > 0:
        metrics['covariance_estimation_time_ms'] = np.mean(covariance_times)
    else:
        metrics['covariance_estimation_time_ms'] = np.nan
    
    if solver_times is not None and len(solver_times) > 0:
        metrics['solver_time_ms'] = np.mean(solver_times)
    else:
        metrics['solver_time_ms'] = np.nan
    
    return metrics

