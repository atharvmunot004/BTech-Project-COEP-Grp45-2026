"""
Metrics computation module for Black-Litterman optimization.

Computes portfolio quality, Black-Litterman specific, risk, structure, 
distribution, comparison, and runtime metrics.
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
    """Compute portfolio quality statistics."""
    assets = portfolio_weights.index.intersection(expected_returns.index)
    assets = assets.intersection(covariance_matrix.index)
    
    if len(assets) == 0:
        return {
            'expected_return': np.nan,
            'volatility': np.nan,
            'portfolio_variance': np.nan
        }
    
    weights = portfolio_weights[assets].values
    mu = expected_returns[assets].values
    Sigma = covariance_matrix.loc[assets, assets].values
    
    expected_return = mu @ weights
    portfolio_variance = weights @ Sigma @ weights
    volatility = np.sqrt(portfolio_variance)
    
    return {
        'expected_return': expected_return,
        'volatility': volatility,
        'portfolio_variance': portfolio_variance
    }


def compute_sharpe_ratio(
    portfolio_returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualize: bool = True
) -> float:
    """Compute Sharpe ratio from portfolio returns."""
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
    """Compute Sortino ratio (downside risk-adjusted return)."""
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
    """Compute maximum drawdown from portfolio returns."""
    returns_clean = portfolio_returns.dropna()
    
    if len(returns_clean) == 0:
        return np.nan
    
    cumulative = (1 + returns_clean).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = abs(drawdown.min())
    return max_drawdown


def compute_calmar_ratio(
    portfolio_returns: pd.Series,
    annualize: bool = True
) -> float:
    """Compute Calmar ratio (annual return / max drawdown)."""
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


def compute_information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    annualize: bool = True
) -> float:
    """Compute information ratio (active return / tracking error)."""
    returns_clean = portfolio_returns.dropna()
    benchmark_clean = benchmark_returns.dropna()
    
    common_index = returns_clean.index.intersection(benchmark_clean.index)
    if len(common_index) < 2:
        return np.nan
    
    active_returns = returns_clean[common_index] - benchmark_clean[common_index]
    tracking_error = active_returns.std()
    
    if annualize:
        tracking_error = tracking_error * np.sqrt(252)
    
    if tracking_error == 0:
        return np.nan
    
    mean_active_return = active_returns.mean()
    if annualize:
        mean_active_return = mean_active_return * 252
    
    info_ratio = mean_active_return / tracking_error
    return info_ratio


def compute_tracking_error(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    annualize: bool = True
) -> float:
    """Compute tracking error (std of active returns)."""
    returns_clean = portfolio_returns.dropna()
    benchmark_clean = benchmark_returns.dropna()
    
    common_index = returns_clean.index.intersection(benchmark_clean.index)
    if len(common_index) < 2:
        return np.nan
    
    active_returns = returns_clean[common_index] - benchmark_clean[common_index]
    tracking_error = active_returns.std()
    
    if annualize:
        tracking_error = tracking_error * np.sqrt(252)
    
    return tracking_error


def compute_alpha_vs_market(
    portfolio_returns: pd.Series,
    market_returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualize: bool = True
) -> float:
    """Compute alpha (excess return) vs market portfolio."""
    returns_clean = portfolio_returns.dropna()
    market_clean = market_returns.dropna()
    
    common_index = returns_clean.index.intersection(market_clean.index)
    if len(common_index) < 2:
        return np.nan
    
    portfolio_mean = returns_clean[common_index].mean()
    market_mean = market_clean[common_index].mean()
    
    if annualize:
        portfolio_mean = portfolio_mean * 252
        market_mean = market_mean * 252
    
    alpha = portfolio_mean - market_mean
    return alpha


def compute_bl_specific_metrics(
    prior_returns: pd.Series,
    posterior_returns: pd.Series,
    market_weights: pd.Series,
    optimal_weights: pd.Series,
    prior_cov: pd.DataFrame,
    posterior_cov: pd.DataFrame,
    P: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
    portfolio_returns: pd.Series,
    market_portfolio_returns: pd.Series,
    risk_free_rate: float,
    tau: float = 0.025,
    risk_aversion: float = 2.5
) -> Dict[str, float]:
    """Compute Black-Litterman specific metrics."""
    metrics = {}
    
    # Prior vs posterior distance
    assets = prior_returns.index.intersection(posterior_returns.index)
    if len(assets) > 0:
        prior_vec = prior_returns[assets].values
        posterior_vec = posterior_returns[assets].values
        metrics['prior_vs_posterior_distance'] = np.linalg.norm(posterior_vec - prior_vec)
        metrics['prior_vs_posterior_correlation'] = np.corrcoef(prior_vec, posterior_vec)[0, 1]
    
    # View consistency
    assets = prior_returns.index.intersection(posterior_returns.index)
    if P.shape[0] > 0 and len(assets) > 0:
        # Use all assets from prior_returns for view consistency check
        prior_vec = prior_returns.values.reshape(-1, 1)
        # P should already be aligned with prior_returns index
        if P.shape[1] == len(prior_returns):
            view_implied = P @ prior_vec
            view_diff = np.abs(Q.flatten() - view_implied.flatten())
            metrics['view_consistency'] = 1.0 / (1.0 + np.mean(view_diff))
        else:
            metrics['view_consistency'] = np.nan
    else:
        metrics['view_consistency'] = np.nan
    
    # View contribution to weights
    if len(assets) > 0:
        market_w = market_weights[assets].values if len(market_weights) > 0 else np.ones(len(assets)) / len(assets)
        optimal_w = optimal_weights[assets].values
        weight_diff = optimal_w - market_w
        metrics['view_contribution_to_weights'] = np.linalg.norm(weight_diff)
        metrics['view_impact_magnitude'] = np.sum(np.abs(weight_diff))
    
    # Market equilibrium weight distance
    if len(assets) > 0:
        market_w = market_weights[assets].values if len(market_weights) > 0 else np.ones(len(assets)) / len(assets)
        optimal_w = optimal_weights[assets].values
        metrics['market_equilibrium_weight_distance'] = np.linalg.norm(optimal_w - market_w)
    
    # Posterior vs prior Sharpe
    prior_sharpe = compute_sharpe_ratio(market_portfolio_returns, risk_free_rate)
    posterior_sharpe = compute_sharpe_ratio(portfolio_returns, risk_free_rate)
    if not np.isnan(prior_sharpe) and not np.isnan(posterior_sharpe):
        metrics['posterior_sharpe_vs_prior_sharpe'] = posterior_sharpe - prior_sharpe
    else:
        metrics['posterior_sharpe_vs_prior_sharpe'] = np.nan
    
    # Information gain from views
    if len(assets) > 0:
        prior_var = np.diag(prior_cov.loc[assets, assets].values)
        posterior_var = np.diag(posterior_cov.loc[assets, assets].values)
        variance_reduction = np.mean(prior_var - posterior_var)
        metrics['information_gain_from_views'] = variance_reduction
    
    # Placeholder metrics
    metrics['tau_sensitivity'] = np.nan
    metrics['risk_aversion_sensitivity'] = np.nan
    metrics['lambda_sensitivity'] = np.nan
    metrics['confidence_level_impact'] = np.nan
    
    # Omega uncertainty impact
    if Omega.shape[0] > 0:
        omega_trace = np.trace(Omega)
        metrics['omega_uncertainty_impact'] = omega_trace
    else:
        metrics['omega_uncertainty_impact'] = np.nan
    
    return metrics


def compute_structure_metrics(
    portfolio_weights: pd.Series,
    covariance_matrix: pd.DataFrame,
    returns: pd.DataFrame,
    market_weights: Optional[pd.Series] = None
) -> Dict[str, float]:
    """Compute portfolio structure metrics."""
    metrics = {}
    
    # Number of assets
    non_zero_weights = portfolio_weights[portfolio_weights > 1e-6]
    metrics['num_assets_in_portfolio'] = len(non_zero_weights)
    
    # HHI concentration
    weights_array = portfolio_weights.values
    hhi = np.sum(weights_array**2)
    metrics['hhi_concentration'] = hhi
    
    # Effective number of assets
    metrics['effective_number_of_assets'] = 1.0 / hhi if hhi > 0 else np.nan
    
    # Weight entropy
    weights_positive = weights_array[weights_array > 1e-10]
    if len(weights_positive) > 0:
        entropy = -np.sum(weights_positive * np.log(weights_positive))
        metrics['weight_entropy'] = entropy
    else:
        metrics['weight_entropy'] = np.nan
    
    # Turnover vs market portfolio
    if market_weights is not None:
        assets = portfolio_weights.index.intersection(market_weights.index)
        if len(assets) > 0:
            portfolio_w = portfolio_weights[assets].values
            market_w = market_weights[assets].values
            turnover = np.sum(np.abs(portfolio_w - market_w)) / 2.0
            metrics['turnover_vs_market_portfolio'] = turnover
            metrics['active_share'] = np.sum(np.abs(portfolio_w - market_w)) / 2.0
        else:
            metrics['turnover_vs_market_portfolio'] = np.nan
            metrics['active_share'] = np.nan
    else:
        metrics['turnover_vs_market_portfolio'] = np.nan
        metrics['active_share'] = np.nan
    
    return metrics


def compute_risk_metrics(portfolio_returns: pd.Series) -> Dict[str, float]:
    """Compute risk metrics."""
    metrics = {}
    
    returns_clean = portfolio_returns.dropna()
    
    if len(returns_clean) < 2:
        return {
            'value_at_risk': np.nan,
            'conditional_value_at_risk': np.nan,
            'downside_deviation': np.nan,
            'posterior_variance': np.nan,
            'prior_variance': np.nan,
            'variance_reduction_from_views': np.nan
        }
    
    # VaR (95% confidence)
    var_95 = np.percentile(returns_clean, 5)
    metrics['value_at_risk'] = abs(var_95)
    
    # CVaR (95% confidence)
    cvar_95 = returns_clean[returns_clean <= var_95].mean()
    metrics['conditional_value_at_risk'] = abs(cvar_95) if not np.isnan(cvar_95) else np.nan
    
    # Downside deviation
    downside_returns = returns_clean[returns_clean < 0]
    if len(downside_returns) > 0:
        metrics['downside_deviation'] = np.sqrt(np.mean(downside_returns**2))
    else:
        metrics['downside_deviation'] = 0.0
    
    # Variance metrics
    metrics['posterior_variance'] = returns_clean.var()
    metrics['prior_variance'] = np.nan
    metrics['variance_reduction_from_views'] = np.nan
    
    return metrics


def compute_distribution_metrics(portfolio_returns: pd.Series) -> Dict[str, float]:
    """Compute distribution metrics and normality checks."""
    metrics = {}
    
    returns_clean = portfolio_returns.dropna()
    
    if len(returns_clean) < 3:
        return {
            'skewness': np.nan,
            'kurtosis': np.nan,
            'jarque_bera_p_value': np.nan
        }
    
    # Skewness
    metrics['skewness'] = stats.skew(returns_clean)
    
    # Kurtosis
    metrics['kurtosis'] = stats.kurtosis(returns_clean)
    
    # Jarque-Bera test
    jb_stat, jb_pvalue = stats.jarque_bera(returns_clean)
    metrics['jarque_bera_p_value'] = jb_pvalue
    
    return metrics


def compute_comparison_metrics(
    prior_portfolio_returns: pd.Series,
    posterior_portfolio_returns: pd.Series,
    market_portfolio_returns: pd.Series,
    prior_returns: pd.Series,
    posterior_returns: pd.Series,
    market_weights: pd.Series,
    optimal_weights: pd.Series,
    risk_free_rate: float
) -> Dict[str, float]:
    """Compute comparison metrics between prior, posterior, and market portfolios."""
    metrics = {}
    
    # Prior portfolio metrics
    if len(prior_portfolio_returns.dropna()) > 0:
        metrics['prior_portfolio_expected_return'] = prior_portfolio_returns.mean() * 252
        metrics['prior_portfolio_volatility'] = prior_portfolio_returns.std() * np.sqrt(252)
        metrics['prior_portfolio_sharpe'] = compute_sharpe_ratio(prior_portfolio_returns, risk_free_rate)
    else:
        metrics['prior_portfolio_expected_return'] = np.nan
        metrics['prior_portfolio_volatility'] = np.nan
        metrics['prior_portfolio_sharpe'] = np.nan
    
    # Posterior portfolio metrics
    if len(posterior_portfolio_returns.dropna()) > 0:
        metrics['posterior_portfolio_expected_return'] = posterior_portfolio_returns.mean() * 252
        metrics['posterior_portfolio_volatility'] = posterior_portfolio_returns.std() * np.sqrt(252)
        metrics['posterior_portfolio_sharpe'] = compute_sharpe_ratio(posterior_portfolio_returns, risk_free_rate)
    else:
        metrics['posterior_portfolio_expected_return'] = np.nan
        metrics['posterior_portfolio_volatility'] = np.nan
        metrics['posterior_portfolio_sharpe'] = np.nan
    
    # Market portfolio metrics
    if len(market_portfolio_returns.dropna()) > 0:
        metrics['market_portfolio_expected_return'] = market_portfolio_returns.mean() * 252
        metrics['market_portfolio_volatility'] = market_portfolio_returns.std() * np.sqrt(252)
        metrics['market_portfolio_sharpe'] = compute_sharpe_ratio(market_portfolio_returns, risk_free_rate)
    else:
        metrics['market_portfolio_expected_return'] = np.nan
        metrics['market_portfolio_volatility'] = np.nan
        metrics['market_portfolio_sharpe'] = np.nan
    
    return metrics


def compute_runtime_metrics(runtimes: list) -> Dict[str, float]:
    """Compute runtime statistics."""
    if len(runtimes) == 0:
        return {}
    
    runtime_array = np.array(runtimes) * 1000  # Convert to ms
    
    return {
        'mean_runtime_ms': float(np.mean(runtime_array)),
        'p95_runtime_ms': float(np.percentile(runtime_array, 95)),
        'median_runtime_ms': float(np.median(runtime_array)),
        'min_runtime_ms': float(np.min(runtime_array)),
        'max_runtime_ms': float(np.max(runtime_array))
    }

