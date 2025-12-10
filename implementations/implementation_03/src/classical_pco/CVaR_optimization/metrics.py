"""
Metrics computation module for CVaR optimization.

Computes CVaR-specific portfolio quality, risk, structure, distribution, and tail risk metrics.
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
import time


def compute_portfolio_statistics(
    portfolio_returns: pd.Series,
    portfolio_weights: pd.Series,
    expected_returns: pd.Series,
    risk_free_rate: float = 0.0
) -> Dict[str, float]:
    """
    Compute basic portfolio quality statistics.
    
    Args:
        portfolio_returns: Series of portfolio returns
        portfolio_weights: Series of portfolio weights
        expected_returns: Series of expected returns
        risk_free_rate: Risk-free rate
        
    Returns:
        Dictionary of portfolio statistics
    """
    returns_clean = portfolio_returns.dropna()
    
    if len(returns_clean) == 0:
        return {
            'expected_return': np.nan,
            'volatility': np.nan
        }
    
    # Expected return
    if len(portfolio_weights) > 0 and len(expected_returns) > 0:
        assets = portfolio_weights.index.intersection(expected_returns.index)
        if len(assets) > 0:
            expected_return = (portfolio_weights[assets] * expected_returns[assets]).sum()
        else:
            expected_return = returns_clean.mean() * 252  # Annualized
    else:
        expected_return = returns_clean.mean() * 252  # Annualized
    
    # Volatility
    volatility = returns_clean.std() * np.sqrt(252)  # Annualized
    
    return {
        'expected_return': expected_return,
        'volatility': volatility
    }


def compute_sharpe_ratio(
    portfolio_returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualize: bool = True
) -> float:
    """Compute Sharpe ratio."""
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
    """Compute Sortino ratio."""
    returns_clean = portfolio_returns.dropna()
    
    if len(returns_clean) < 2:
        return np.nan
    
    mean_return = returns_clean.mean()
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
    """Compute maximum drawdown."""
    returns_clean = portfolio_returns.dropna()
    
    if len(returns_clean) == 0:
        return np.nan
    
    cumulative = (1 + returns_clean).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = abs(drawdown.min())
    return max_drawdown


def compute_calmar_ratio(portfolio_returns: pd.Series, annualize: bool = True) -> float:
    """Compute Calmar ratio."""
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


def compute_var_cvar(
    portfolio_returns: pd.Series,
    confidence_level: float = 0.95,
    method: str = 'historical'
) -> Tuple[float, float]:
    """
    Compute Value at Risk (VaR) and Conditional Value at Risk (CVaR).
    
    Args:
        portfolio_returns: Series of portfolio returns
        confidence_level: Confidence level alpha
        method: 'historical' or 'parametric'
        
    Returns:
        Tuple of (VaR, CVaR)
    """
    returns_clean = portfolio_returns.dropna()
    
    if len(returns_clean) == 0:
        return np.nan, np.nan
    
    alpha = 1 - confidence_level
    
    if method == 'historical':
        # Historical VaR
        var_value = -np.percentile(returns_clean, alpha * 100)
        
        # Historical CVaR (expected shortfall)
        var_threshold = -var_value
        tail_returns = returns_clean[returns_clean <= var_threshold]
        if len(tail_returns) > 0:
            cvar_value = -tail_returns.mean()
        else:
            cvar_value = var_value
    else:
        # Parametric method
        mean_return = returns_clean.mean()
        std_return = returns_clean.std()
        z_score = stats.norm.ppf(confidence_level)
        var_value = -(mean_return - z_score * std_return)
        cvar_value = -(mean_return - (stats.norm.pdf(z_score) / alpha) * std_return)
    
    return var_value, cvar_value


def compute_cvar_sharpe_ratio(
    portfolio_returns: pd.Series,
    risk_free_rate: float = 0.0,
    confidence_level: float = 0.95,
    annualize: bool = True
) -> float:
    """Compute CVaR-adjusted Sharpe ratio."""
    returns_clean = portfolio_returns.dropna()
    
    if len(returns_clean) == 0:
        return np.nan
    
    mean_return = returns_clean.mean()
    _, cvar = compute_var_cvar(returns_clean, confidence_level, method='historical')
    
    if annualize:
        mean_return = mean_return * 252
        cvar = cvar * np.sqrt(252)
    
    if cvar == 0 or np.isnan(cvar):
        return np.nan
    
    cvar_sharpe = (mean_return - risk_free_rate) / cvar
    return cvar_sharpe


def compute_cvar_sortino_ratio(
    portfolio_returns: pd.Series,
    risk_free_rate: float = 0.0,
    confidence_level: float = 0.95,
    annualize: bool = True
) -> float:
    """Compute CVaR-adjusted Sortino ratio."""
    returns_clean = portfolio_returns.dropna()
    
    if len(returns_clean) == 0:
        return np.nan
    
    mean_return = returns_clean.mean()
    _, cvar = compute_var_cvar(returns_clean, confidence_level, method='historical')
    
    if annualize:
        mean_return = mean_return * 252
        cvar = cvar * np.sqrt(252)
    
    if cvar == 0 or np.isnan(cvar):
        return np.nan
    
    cvar_sortino = (mean_return - risk_free_rate) / cvar
    return cvar_sortino


def compute_cvar_calmar_ratio(
    portfolio_returns: pd.Series,
    confidence_level: float = 0.95,
    annualize: bool = True
) -> float:
    """Compute CVaR-adjusted Calmar ratio."""
    returns_clean = portfolio_returns.dropna()
    
    if len(returns_clean) == 0:
        return np.nan
    
    mean_return = returns_clean.mean()
    _, cvar = compute_var_cvar(returns_clean, confidence_level, method='historical')
    
    if annualize:
        mean_return = mean_return * 252
        cvar = cvar * np.sqrt(252)
    
    max_dd = compute_max_drawdown(portfolio_returns)
    
    if max_dd == 0 or np.isnan(max_dd) or np.isnan(cvar):
        return np.nan
    
    cvar_calmar = mean_return / (cvar + max_dd)  # Combined risk measure
    return cvar_calmar


def compute_return_over_cvar(
    portfolio_returns: pd.Series,
    confidence_level: float = 0.95,
    annualize: bool = True
) -> float:
    """Compute return over CVaR ratio."""
    returns_clean = portfolio_returns.dropna()
    
    if len(returns_clean) == 0:
        return np.nan
    
    mean_return = returns_clean.mean()
    _, cvar = compute_var_cvar(returns_clean, confidence_level, method='historical')
    
    if annualize:
        mean_return = mean_return * 252
        cvar = cvar * np.sqrt(252)
    
    if cvar == 0 or np.isnan(cvar):
        return np.nan
    
    return mean_return / cvar


def compute_return_over_var(
    portfolio_returns: pd.Series,
    confidence_level: float = 0.95,
    annualize: bool = True
) -> float:
    """Compute return over VaR ratio."""
    returns_clean = portfolio_returns.dropna()
    
    if len(returns_clean) == 0:
        return np.nan
    
    mean_return = returns_clean.mean()
    var, _ = compute_var_cvar(returns_clean, confidence_level, method='historical')
    
    if annualize:
        mean_return = mean_return * 252
        var = var * np.sqrt(252)
    
    if var == 0 or np.isnan(var):
        return np.nan
    
    return mean_return / var


def compute_risk_metrics(
    portfolio_returns: pd.Series,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Compute comprehensive risk metrics.
    
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
            'expected_shortfall': np.nan,
            'downside_deviation': np.nan,
            'semivariance': np.nan,
            'tail_ratio': np.nan,
            'cvar_var_ratio': np.nan,
            'worst_case_loss': np.nan,
            'expected_loss_beyond_cvar': np.nan
        }
    
    # VaR and CVaR
    var, cvar = compute_var_cvar(returns_clean, confidence_level, method='historical')
    
    # Expected shortfall (same as CVaR)
    expected_shortfall = cvar
    
    # Downside deviation
    downside_returns = returns_clean[returns_clean < 0]
    downside_deviation = np.sqrt(np.mean(downside_returns**2)) if len(downside_returns) > 0 else 0.0
    
    # Semivariance (downside variance)
    semivariance = np.var(downside_returns) if len(downside_returns) > 0 else 0.0
    
    # Tail ratio (95th percentile / 5th percentile)
    tail_ratio = abs(np.percentile(returns_clean, 95) / np.percentile(returns_clean, 5)) if np.percentile(returns_clean, 5) != 0 else np.nan
    
    # CVaR/VaR ratio
    cvar_var_ratio = cvar / var if var != 0 and not np.isnan(var) else np.nan
    
    # Worst case loss (minimum return)
    worst_case_loss = returns_clean.min()
    
    # Expected loss beyond CVaR (extreme tail)
    var_threshold = -var
    extreme_tail = returns_clean[returns_clean < var_threshold]
    if len(extreme_tail) > 0:
        expected_loss_beyond_cvar = -extreme_tail.mean() - cvar
    else:
        expected_loss_beyond_cvar = 0.0
    
    return {
        'value_at_risk': var,
        'conditional_value_at_risk': cvar,
        'expected_shortfall': expected_shortfall,
        'downside_deviation': downside_deviation,
        'semivariance': semivariance,
        'tail_ratio': tail_ratio,
        'cvar_var_ratio': cvar_var_ratio,
        'worst_case_loss': worst_case_loss,
        'expected_loss_beyond_cvar': expected_loss_beyond_cvar
    }


def compute_cvar_sensitivity(
    portfolio_returns: pd.Series,
    confidence_levels: List[float] = [0.90, 0.95, 0.99]
) -> Dict[str, float]:
    """
    Compute CVaR sensitivity to confidence level.
    
    Args:
        portfolio_returns: Series of portfolio returns
        confidence_levels: List of confidence levels to test
        
    Returns:
        Dictionary with CVaR values at different confidence levels
    """
    sensitivity = {}
    
    for conf_level in confidence_levels:
        _, cvar = compute_var_cvar(portfolio_returns, conf_level, method='historical')
        sensitivity[f'cvar_{int(conf_level*100)}'] = cvar
    
    return sensitivity


def compute_var_sensitivity(
    portfolio_returns: pd.Series,
    confidence_levels: List[float] = [0.90, 0.95, 0.99]
) -> Dict[str, float]:
    """
    Compute VaR sensitivity to confidence level.
    
    Args:
        portfolio_returns: Series of portfolio returns
        confidence_levels: List of confidence levels to test
        
    Returns:
        Dictionary with VaR values at different confidence levels
    """
    sensitivity = {}
    
    for conf_level in confidence_levels:
        var, _ = compute_var_cvar(portfolio_returns, conf_level, method='historical')
        sensitivity[f'var_{int(conf_level*100)}'] = var
    
    return sensitivity


def compute_structure_metrics(
    portfolio_weights: pd.Series,
    returns: Optional[pd.DataFrame] = None
) -> Dict[str, float]:
    """
    Compute portfolio structure metrics.
    
    Args:
        portfolio_weights: Series of portfolio weights
        returns: Optional returns DataFrame for correlation metrics
        
    Returns:
        Dictionary of structure metrics
    """
    weights_array = portfolio_weights.values
    active_weights = weights_array[weights_array > 1e-10]
    
    num_assets = len(active_weights)
    
    # HHI
    hhi = np.sum(weights_array**2)
    
    # Effective number of assets
    effective_num_assets = 1 / hhi if hhi > 0 else np.nan
    
    # Weight entropy
    weights_normalized = weights_array[weights_array > 1e-10]
    if len(weights_normalized) > 0:
        weights_normalized = weights_normalized / weights_normalized.sum()
        weight_entropy = -np.sum(weights_normalized * np.log(weights_normalized + 1e-10))
    else:
        weight_entropy = np.nan
    
    # Pairwise correlation mean
    pairwise_correlation_mean = np.nan
    if returns is not None:
        try:
            assets = portfolio_weights.index.intersection(returns.columns)
            if len(assets) > 1:
                corr_matrix = returns[assets].corr().values
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                pairwise_correlations = corr_matrix[mask]
                pairwise_correlation_mean = np.mean(pairwise_correlations)
        except Exception:
            pass
    
    return {
        'num_assets_in_portfolio': num_assets,
        'hhi_concentration': hhi,
        'effective_number_of_assets': effective_num_assets,
        'weight_entropy': weight_entropy,
        'pairwise_correlation_mean': pairwise_correlation_mean
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
    kurtosis = stats.kurtosis(returns_clean)
    jb_stat, jb_p_value = stats.jarque_bera(returns_clean)
    
    return {
        'skewness': skewness,
        'kurtosis': kurtosis,
        'jarque_bera_p_value': jb_p_value
    }


def compute_tail_risk_metrics(
    portfolio_returns: pd.Series,
    confidence_levels: List[float] = [0.95, 0.99]
) -> Dict[str, float]:
    """
    Compute tail risk analysis metrics.
    
    Args:
        portfolio_returns: Series of portfolio returns
        confidence_levels: List of confidence levels
        
    Returns:
        Dictionary of tail risk metrics
    """
    returns_clean = portfolio_returns.dropna()
    
    if len(returns_clean) < 10:
        return {
            'tail_index': np.nan,
            'extreme_value_risk': np.nan,
            'cvar_coherence_check': np.nan,
            'cvar_consistency_across_conf_levels': np.nan
        }
    
    # Tail index (Pareto tail index estimation)
    # Simple estimation using Hill estimator
    sorted_returns = np.sort(returns_clean)
    n_tail = max(10, len(sorted_returns) // 10)  # Use top 10% for tail
    tail_returns = sorted_returns[:n_tail]
    
    if len(tail_returns) > 1 and np.min(tail_returns) < 0:
        # Hill estimator for left tail
        log_ratios = np.log(-tail_returns / (-tail_returns[-1] + 1e-10))
        tail_index = 1 / np.mean(log_ratios) if np.mean(log_ratios) > 0 else np.nan
    else:
        tail_index = np.nan
    
    # Extreme value risk (worst 1% expected loss)
    extreme_risk = -np.percentile(returns_clean, 1)
    
    # CVaR coherence check (CVaR should be >= VaR)
    var_95, cvar_95 = compute_var_cvar(returns_clean, 0.95, method='historical')
    coherence_check = 1.0 if cvar_95 >= var_95 else 0.0
    
    # CVaR consistency across confidence levels
    cvar_values = []
    for conf_level in confidence_levels:
        _, cvar = compute_var_cvar(returns_clean, conf_level, method='historical')
        cvar_values.append(cvar)
    
    # Check if CVaR increases with confidence level (should be monotonic)
    if len(cvar_values) > 1:
        consistency = 1.0 if all(cvar_values[i] >= cvar_values[i-1] for i in range(1, len(cvar_values))) else 0.0
    else:
        consistency = np.nan
    
    return {
        'tail_index': tail_index,
        'extreme_value_risk': extreme_risk,
        'cvar_coherence_check': coherence_check,
        'cvar_consistency_across_conf_levels': consistency
    }


def compute_runtime_metrics(
    runtimes: List[float],
    scenario_times: Optional[List[float]] = None,
    solver_times: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Compute runtime performance metrics.
    
    Args:
        runtimes: List of total runtime values in seconds
        scenario_times: List of scenario construction times in ms
        solver_times: List of solver times in ms
        
    Returns:
        Dictionary of runtime metrics
    """
    metrics = {}
    
    if len(runtimes) > 0:
        runtimes_array = np.array(runtimes) * 1000  # Convert to ms
        metrics['runtime_per_optimization_ms'] = float(np.mean(runtimes_array))
        metrics['p95_runtime_ms'] = float(np.percentile(runtimes_array, 95))
    else:
        metrics['runtime_per_optimization_ms'] = np.nan
        metrics['p95_runtime_ms'] = np.nan
    
    if scenario_times is not None and len(scenario_times) > 0:
        metrics['scenario_construction_time_ms'] = float(np.mean(scenario_times))
    else:
        metrics['scenario_construction_time_ms'] = np.nan
    
    if solver_times is not None and len(solver_times) > 0:
        metrics['solver_time_ms'] = float(np.mean(solver_times))
    else:
        metrics['solver_time_ms'] = np.nan
    
    return metrics

