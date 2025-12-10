"""
Metrics computation module for Risk Parity ERC optimization.

Computes portfolio quality, Risk Parity specific, risk, structure, 
distribution, comparison, and runtime metrics.
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Optional, Tuple
import time

from .risk_parity_erc_optimizer import calculate_risk_contributions


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


def compute_risk_parity_specific_metrics(
    portfolio_weights: pd.Series,
    covariance_matrix: pd.DataFrame
) -> Dict[str, float]:
    """
    Compute Risk Parity / ERC specific metrics.
    
    Args:
        portfolio_weights: Portfolio weights Series
        covariance_matrix: Covariance matrix DataFrame
        
    Returns:
        Dictionary of ERC-specific metrics
    """
    metrics = {}
    
    weights = portfolio_weights.values
    n = len(weights)
    
    # Calculate risk contributions
    risk_contrib, marginal_contrib, portfolio_vol = calculate_risk_contributions(
        weights, covariance_matrix
    )
    
    # Risk contribution vector (as a summary statistic - mean)
    metrics['risk_contribution_vector'] = np.mean(risk_contrib) if len(risk_contrib) > 0 else np.nan
    
    # Risk contribution variance
    metrics['risk_contribution_variance'] = np.var(risk_contrib) if len(risk_contrib) > 0 else np.nan
    
    # Risk contribution standard deviation
    metrics['risk_contribution_std'] = np.std(risk_contrib) if len(risk_contrib) > 0 else np.nan
    
    # Risk contribution coefficient of variation
    mean_rc = np.mean(risk_contrib) if len(risk_contrib) > 0 else 0
    if mean_rc > 0:
        metrics['risk_contribution_coefficient_of_variation'] = np.std(risk_contrib) / mean_rc
    else:
        metrics['risk_contribution_coefficient_of_variation'] = np.nan
    
    # Max and min risk contributions
    if len(risk_contrib) > 0:
        metrics['max_risk_contribution'] = np.max(risk_contrib)
        metrics['min_risk_contribution'] = np.min(risk_contrib)
    else:
        metrics['max_risk_contribution'] = np.nan
        metrics['min_risk_contribution'] = np.nan
    
    # Equal risk gap (deviation from perfect equality)
    if portfolio_vol > 0 and n > 0:
        target_rc = portfolio_vol / n
        equal_risk_gap = np.mean(np.abs(risk_contrib - target_rc))
        metrics['equal_risk_gap'] = equal_risk_gap
    else:
        metrics['equal_risk_gap'] = np.nan
    
    # Volatility contribution per asset (same as risk contribution for volatility-based ERC)
    metrics['volatility_contribution_per_asset'] = np.mean(risk_contrib) if len(risk_contrib) > 0 else np.nan
    
    # Marginal risk contribution vector (as summary - mean)
    metrics['marginal_risk_contribution_vector'] = np.mean(marginal_contrib) if len(marginal_contrib) > 0 else np.nan
    
    # Risk parity deviation score (normalized measure of how far from perfect ERC)
    if portfolio_vol > 0 and n > 0:
        target_rc = portfolio_vol / n
        if target_rc > 0:
            deviation_score = np.sqrt(np.mean((risk_contrib - target_rc)**2)) / target_rc
            metrics['risk_parity_deviation_score'] = deviation_score
        else:
            metrics['risk_parity_deviation_score'] = np.nan
    else:
        metrics['risk_parity_deviation_score'] = np.nan
    
    # Risk concentration index (coefficient of variation of risk contributions)
    if mean_rc > 0:
        metrics['risk_concentration_index'] = np.std(risk_contrib) / mean_rc
    else:
        metrics['risk_concentration_index'] = np.nan
    
    return metrics


def compute_structure_metrics(
    portfolio_weights: pd.Series,
    covariance_matrix: pd.DataFrame,
    returns: pd.DataFrame,
    baseline_weights: Optional[pd.Series] = None
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
    
    # Pairwise correlation mean
    if len(returns.columns) > 1:
        corr_matrix = returns.corr().values
        # Get upper triangle (excluding diagonal)
        upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        metrics['pairwise_correlation_mean'] = np.mean(upper_triangle) if len(upper_triangle) > 0 else np.nan
    else:
        metrics['pairwise_correlation_mean'] = np.nan
    
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
            'semivariance': np.nan
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
    
    # Semivariance (variance of negative returns)
    negative_returns = returns_clean[returns_clean < 0]
    if len(negative_returns) > 0:
        metrics['semivariance'] = np.var(negative_returns)
    else:
        metrics['semivariance'] = 0.0
    
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
    erc_portfolio_returns: pd.Series,
    baseline_portfolio_returns: Optional[pd.Series],
    erc_weights: pd.Series,
    baseline_weights: Optional[pd.Series],
    erc_covariance: pd.DataFrame,
    baseline_covariance: Optional[pd.DataFrame],
    risk_free_rate: float,
    returns_dataframe: Optional[pd.DataFrame] = None
) -> Dict[str, float]:
    """Compute comparison metrics between ERC and baseline portfolios."""
    metrics = {}
    
    # Baseline portfolio metrics
    if baseline_portfolio_returns is not None and len(baseline_portfolio_returns.dropna()) > 0:
        metrics['baseline_portfolio_volatility'] = baseline_portfolio_returns.std() * np.sqrt(252)
        metrics['baseline_portfolio_sharpe'] = compute_sharpe_ratio(baseline_portfolio_returns, risk_free_rate)
        metrics['baseline_portfolio_expected_return'] = baseline_portfolio_returns.mean() * 252
    else:
        metrics['baseline_portfolio_volatility'] = np.nan
        metrics['baseline_portfolio_sharpe'] = np.nan
        metrics['baseline_portfolio_expected_return'] = np.nan
    
    # ERC portfolio metrics
    erc_vol = erc_portfolio_returns.std() * np.sqrt(252) if len(erc_portfolio_returns.dropna()) > 0 else np.nan
    baseline_vol = metrics['baseline_portfolio_volatility']
    
    # Volatility reduction vs baseline
    if not np.isnan(erc_vol) and not np.isnan(baseline_vol) and baseline_vol > 0:
        metrics['volatility_reduction_vs_baseline'] = (baseline_vol - erc_vol) / baseline_vol
    else:
        metrics['volatility_reduction_vs_baseline'] = np.nan
    
    # Sharpe improvement vs baseline
    erc_sharpe = compute_sharpe_ratio(erc_portfolio_returns, risk_free_rate)
    baseline_sharpe = metrics['baseline_portfolio_sharpe']
    if not np.isnan(erc_sharpe) and not np.isnan(baseline_sharpe):
        metrics['sharpe_improvement_vs_baseline'] = erc_sharpe - baseline_sharpe
    else:
        metrics['sharpe_improvement_vs_baseline'] = np.nan
    
    # Risk contribution improvement vs baseline
    erc_rc, _, _ = calculate_risk_contributions(erc_weights.values, erc_covariance)
    if baseline_weights is not None and baseline_covariance is not None:
        baseline_rc, _, _ = calculate_risk_contributions(baseline_weights.values, baseline_covariance)
        if len(erc_rc) > 0 and len(baseline_rc) > 0:
            erc_rc_std = np.std(erc_rc)
            baseline_rc_std = np.std(baseline_rc)
            if baseline_rc_std > 0:
                metrics['risk_contribution_improvement_vs_baseline'] = (baseline_rc_std - erc_rc_std) / baseline_rc_std
            else:
                metrics['risk_contribution_improvement_vs_baseline'] = np.nan
        else:
            metrics['risk_contribution_improvement_vs_baseline'] = np.nan
    else:
        metrics['risk_contribution_improvement_vs_baseline'] = np.nan
    
    # ERC vs equal weight comparison
    n = len(erc_weights)
    equal_weights = pd.Series(np.ones(n) / n, index=erc_weights.index)
    eq_weights = equal_weights.values
    eq_cov = erc_covariance.loc[equal_weights.index, equal_weights.index].values
    eq_vol = np.sqrt(eq_weights @ eq_cov @ eq_weights) * np.sqrt(252)
    
    # For Sharpe, compute from returns if available
    if returns_dataframe is not None and len(returns_dataframe) > 0:
        # Get assets that match
        assets = erc_weights.index.intersection(returns_dataframe.columns)
        if len(assets) > 0:
            equal_weights_subset = equal_weights[assets] / equal_weights[assets].sum()
            equal_weight_returns = (returns_dataframe[assets] * equal_weights_subset).sum(axis=1)
            eq_sharpe = compute_sharpe_ratio(equal_weight_returns, risk_free_rate)
        else:
            eq_sharpe = np.nan
    else:
        eq_sharpe = np.nan
    
    metrics['erc_vs_equal_weight_volatility'] = erc_vol - eq_vol if not np.isnan(erc_vol) and not np.isnan(eq_vol) else np.nan
    metrics['erc_vs_equal_weight_sharpe'] = erc_sharpe - eq_sharpe if not np.isnan(erc_sharpe) and not np.isnan(eq_sharpe) else np.nan
    
    # ERC vs equal weight risk contributions
    eq_rc, _, _ = calculate_risk_contributions(equal_weights.values, erc_covariance)
    if len(erc_rc) > 0 and len(eq_rc) > 0:
        metrics['erc_vs_equal_weight_risk_contributions'] = np.mean(np.abs(erc_rc - eq_rc))
    else:
        metrics['erc_vs_equal_weight_risk_contributions'] = np.nan
    
    # Difference in risk contributions vs baseline
    if baseline_weights is not None and baseline_covariance is not None:
        baseline_rc, _, _ = calculate_risk_contributions(baseline_weights.values, baseline_covariance)
        if len(erc_rc) > 0 and len(baseline_rc) > 0:
            min_len = min(len(erc_rc), len(baseline_rc))
            metrics['difference_in_risk_contributions_vs_baseline'] = np.mean(np.abs(erc_rc[:min_len] - baseline_rc[:min_len]))
        else:
            metrics['difference_in_risk_contributions_vs_baseline'] = np.nan
    else:
        metrics['difference_in_risk_contributions_vs_baseline'] = np.nan
    
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

