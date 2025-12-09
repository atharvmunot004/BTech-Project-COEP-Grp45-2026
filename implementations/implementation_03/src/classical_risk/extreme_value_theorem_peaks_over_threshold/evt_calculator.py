"""
EVT-POT calculator for VaR and CVaR estimation using Extreme Value Theory.

Implements Peaks Over Threshold (POT) methodology with Generalized Pareto Distribution (GPD)
for estimating extreme risk measures.
"""
import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple, Dict, Any
import warnings
import time
from scipy import stats
from scipy.optimize import minimize

try:
    from scipy.stats import genpareto
    SCIPY_GPD_AVAILABLE = True
except ImportError:
    SCIPY_GPD_AVAILABLE = False
    warnings.warn("scipy.stats.genpareto not available. GPD functionality may be limited.")


def select_threshold_quantile(
    returns: pd.Series,
    quantile: float = 0.95,
    min_exceedances: int = 50
) -> Tuple[float, int]:
    """
    Select threshold using quantile method.
    
    Args:
        returns: Series of returns (negative returns for losses)
        quantile: Quantile level for threshold (e.g., 0.95 for 95th percentile)
        min_exceedances: Minimum number of exceedances required
        
    Returns:
        Tuple of (threshold, number_of_exceedances)
    """
    # For EVT, we work with negative returns (losses)
    losses = -returns
    
    # Calculate threshold as quantile
    threshold = losses.quantile(quantile)
    
    # Count exceedances
    exceedances = losses[losses > threshold]
    num_exceedances = len(exceedances)
    
    # If not enough exceedances, try lower quantile
    if num_exceedances < min_exceedances:
        # Try progressively lower quantiles
        for q in [0.90, 0.85, 0.80, 0.75, 0.70]:
            threshold = losses.quantile(q)
            exceedances = losses[losses > threshold]
            num_exceedances = len(exceedances)
            if num_exceedances >= min_exceedances:
                break
    
    return float(threshold), num_exceedances


def select_threshold_automatic(
    returns: pd.Series,
    quantiles: list = [0.90, 0.95, 0.97],
    min_exceedances: int = 50
) -> Tuple[float, int, Dict]:
    """
    Automatically select threshold using mean excess plot or stability criteria.
    
    Args:
        returns: Series of returns (negative returns for losses)
        quantiles: List of quantiles to try
        min_exceedances: Minimum number of exceedances required
        
    Returns:
        Tuple of (threshold, number_of_exceedances, diagnostics_dict)
    """
    losses = -returns
    
    best_threshold = None
    best_num_exceedances = 0
    best_quantile = None
    diagnostics = {}
    
    # Try each quantile
    for quantile in sorted(quantiles, reverse=True):
        threshold, num_exceedances = select_threshold_quantile(
            returns, quantile, min_exceedances
        )
        
        if num_exceedances >= min_exceedances:
            # Calculate mean excess for this threshold
            exceedances = losses[losses > threshold]
            mean_excess = exceedances.mean() - threshold if len(exceedances) > 0 else np.nan
            
            diagnostics[quantile] = {
                'threshold': threshold,
                'num_exceedances': num_exceedances,
                'mean_excess': mean_excess
            }
            
            # Prefer threshold with more exceedances (but not too many)
            if num_exceedances > best_num_exceedances:
                best_threshold = threshold
                best_num_exceedances = num_exceedances
                best_quantile = quantile
    
    # If no threshold found, use the highest quantile that gives at least some exceedances
    if best_threshold is None:
        for quantile in sorted(quantiles, reverse=True):
            threshold, num_exceedances = select_threshold_quantile(
                returns, quantile, 10  # Lower minimum
            )
            if num_exceedances >= 10:
                best_threshold = threshold
                best_num_exceedances = num_exceedances
                best_quantile = quantile
                break
    
    if best_threshold is None:
        # Fallback: use 90th percentile
        best_threshold, best_num_exceedances = select_threshold_quantile(returns, 0.90, 10)
        best_quantile = 0.90
    
    diagnostics['selected_quantile'] = best_quantile
    diagnostics['selected_threshold'] = best_threshold
    diagnostics['selected_exceedances'] = best_num_exceedances
    
    return float(best_threshold), best_num_exceedances, diagnostics


def extract_exceedances(
    returns: pd.Series,
    threshold: float
) -> pd.Series:
    """
    Extract exceedances over threshold.
    
    Args:
        returns: Series of returns
        threshold: Threshold value
        
    Returns:
        Series of exceedances (losses - threshold)
    """
    losses = -returns
    exceedances = losses[losses > threshold] - threshold
    
    return exceedances


def fit_gpd_mle(
    exceedances: pd.Series,
    xi_lower: float = -0.5,
    xi_upper: float = 0.5
) -> Tuple[float, float, Dict]:
    """
    Fit Generalized Pareto Distribution using Maximum Likelihood Estimation.
    
    Args:
        exceedances: Series of exceedances (losses - threshold)
        xi_lower: Lower bound for shape parameter xi
        xi_upper: Upper bound for shape parameter xi
        
    Returns:
        Tuple of (xi (shape), beta (scale), diagnostics_dict)
    """
    if len(exceedances) == 0:
        return np.nan, np.nan, {'error': 'No exceedances'}
    
    exceedances_array = exceedances.values
    
    # Initial parameter estimates using method of moments
    mean_exceedance = exceedances_array.mean()
    var_exceedance = exceedances_array.var()
    
    if var_exceedance <= 0 or mean_exceedance <= 0:
        # Fallback: use simple estimates
        xi_init = 0.1
        beta_init = mean_exceedance
    else:
        # Method of moments estimates
        xi_init = 0.5 * (1 - (mean_exceedance**2 / var_exceedance))
        beta_init = 0.5 * mean_exceedance * (1 + (mean_exceedance**2 / var_exceedance))
        
        # Constrain initial values
        xi_init = np.clip(xi_init, xi_lower + 0.01, xi_upper - 0.01)
        beta_init = max(beta_init, 0.001)
    
    # Negative log-likelihood function
    def neg_log_likelihood(params):
        xi, beta = params
        
        # Constrain parameters
        if beta <= 0:
            return 1e10
        if xi < xi_lower or xi > xi_upper:
            return 1e10
        
        # Handle different cases for GPD
        if abs(xi) < 1e-8:  # Exponential case (xi = 0)
            log_likelihood = -len(exceedances_array) * np.log(beta) - exceedances_array.sum() / beta
        else:
            # General case
            z = 1 + xi * exceedances_array / beta
            if np.any(z <= 0):
                return 1e10
            log_likelihood = (
                -len(exceedances_array) * np.log(beta) -
                (1 + 1/xi) * np.sum(np.log(z))
            )
        
        return -log_likelihood
    
    # Optimize
    try:
        result = minimize(
            neg_log_likelihood,
            x0=[xi_init, beta_init],
            method='L-BFGS-B',
            bounds=[(xi_lower, xi_upper), (0.001, None)],
            options={'maxiter': 1000}
        )
        
        if result.success:
            xi_est = result.x[0]
            beta_est = result.x[1]
            
            # Calculate standard errors (approximate)
            try:
                # Use numerical Hessian for standard errors
                from scipy.optimize import approx_fprime
                hessian = np.linalg.inv(
                    np.array([
                        [approx_fprime([xi_est, beta_est], lambda p: neg_log_likelihood([p[0], beta_est]), 1e-6)[0],
                         approx_fprime([xi_est, beta_est], lambda p: neg_log_likelihood([xi_est, p[1]]), 1e-6)[0]],
                        [approx_fprime([xi_est, beta_est], lambda p: neg_log_likelihood([p[0], beta_est]), 1e-6)[1],
                         approx_fprime([xi_est, beta_est], lambda p: neg_log_likelihood([xi_est, p[1]]), 1e-6)[1]]
                    ])
                )
                se_xi = np.sqrt(hessian[0, 0]) if hessian[0, 0] > 0 else np.nan
                se_beta = np.sqrt(hessian[1, 1]) if hessian[1, 1] > 0 else np.nan
            except:
                se_xi = np.nan
                se_beta = np.nan
            
            diagnostics = {
                'success': True,
                'xi': float(xi_est),
                'beta': float(beta_est),
                'se_xi': float(se_xi) if not np.isnan(se_xi) else None,
                'se_beta': float(se_beta) if not np.isnan(se_beta) else None,
                'log_likelihood': float(-result.fun),
                'num_exceedances': len(exceedances_array)
            }
        else:
            # Fallback to method of moments
            xi_est = xi_init
            beta_est = beta_init
            diagnostics = {
                'success': False,
                'xi': float(xi_est),
                'beta': float(beta_est),
                'method': 'method_of_moments',
                'num_exceedances': len(exceedances_array)
            }
    except Exception as e:
        # Fallback to method of moments
        xi_est = xi_init
        beta_est = beta_init
        diagnostics = {
            'success': False,
            'xi': float(xi_est),
            'beta': float(beta_est),
            'method': 'method_of_moments',
            'error': str(e),
            'num_exceedances': len(exceedances_array)
        }
    
    return float(xi_est), float(beta_est), diagnostics


def compute_var_from_evt(
    returns: pd.Series,
    threshold: float,
    xi: float,
    beta: float,
    confidence_level: float = 0.95,
    horizon: int = 1
) -> float:
    """
    Compute VaR from EVT-POT using GPD parameters.
    
    Args:
        returns: Series of returns
        threshold: Threshold used for POT
        xi: GPD shape parameter
        beta: GPD scale parameter
        confidence_level: Confidence level (e.g., 0.95)
        horizon: Time horizon in days
        
    Returns:
        VaR value
    """
    losses = -returns
    n = len(losses)
    nu = len(losses[losses > threshold])  # Number of exceedances
    
    if nu == 0:
        return np.nan
    
    # Probability of exceedance
    p_exceed = nu / n
    
    # Target probability for VaR
    p_target = 1 - confidence_level
    
    # Adjust for horizon (simple scaling)
    p_target_horizon = 1 - (confidence_level ** (1.0 / horizon))
    
    if p_target_horizon <= p_exceed:
        # VaR is above threshold
        if abs(xi) < 1e-8:  # Exponential case
            var = threshold + beta * np.log(p_exceed / p_target_horizon)
        else:
            var = threshold + (beta / xi) * (((p_exceed / p_target_horizon) ** xi) - 1)
    else:
        # VaR is below threshold, use empirical quantile
        var = losses.quantile(1 - p_target_horizon)
    
    return float(var)


def compute_cvar_from_evt(
    returns: pd.Series,
    threshold: float,
    xi: float,
    beta: float,
    var_value: float,
    confidence_level: float = 0.95,
    horizon: int = 1
) -> float:
    """
    Compute CVaR (Expected Shortfall) from EVT-POT using GPD parameters.
    
    Args:
        returns: Series of returns
        threshold: Threshold used for POT
        xi: GPD shape parameter
        beta: GPD scale parameter
        var_value: VaR value (already computed)
        confidence_level: Confidence level (e.g., 0.95)
        horizon: Time horizon in days
        
    Returns:
        CVaR value
    """
    if np.isnan(var_value) or var_value <= threshold:
        # If VaR is below threshold, use empirical CVaR
        losses = -returns
        p_target = 1 - confidence_level
        p_target_horizon = 1 - (confidence_level ** (1.0 / horizon))
        cvar = losses[losses >= losses.quantile(1 - p_target_horizon)].mean()
        return float(cvar)
    
    # CVaR for GPD
    if abs(xi) < 1e-8:  # Exponential case
        cvar = var_value + beta
    elif xi < 1:  # Finite mean case
        cvar = var_value + (beta - xi * (var_value - threshold)) / (1 - xi)
    else:
        # Infinite mean case
        cvar = np.inf
    
    return float(cvar)


def compute_rolling_var_from_evt(
    returns: pd.Series,
    window: int = 252,
    confidence_level: float = 0.95,
    horizon: int = 1,
    threshold_quantiles: list = [0.90, 0.95, 0.97],
    min_exceedances: int = 50,
    xi_lower: float = -0.5,
    xi_upper: float = 0.5,
    automatic_threshold: bool = True
) -> pd.Series:
    """
    Compute rolling VaR using EVT-POT methodology.
    
    Args:
        returns: Series of returns
        window: Rolling window size
        confidence_level: Confidence level
        horizon: Time horizon in days
        threshold_quantiles: List of quantiles to try for threshold
        min_exceedances: Minimum number of exceedances
        xi_lower: Lower bound for shape parameter
        xi_upper: Upper bound for shape parameter
        automatic_threshold: If True, automatically select threshold
        
    Returns:
        Series of VaR values
    """
    var_series = pd.Series(index=returns.index, dtype=float)
    
    for i in range(window, len(returns) + 1):
        window_returns = returns.iloc[i-window:i]
        
        try:
            # Select threshold
            if automatic_threshold:
                threshold, num_exceedances, _ = select_threshold_automatic(
                    window_returns,
                    threshold_quantiles,
                    min_exceedances
                )
            else:
                threshold, num_exceedances = select_threshold_quantile(
                    window_returns,
                    threshold_quantiles[0] if threshold_quantiles else 0.95,
                    min_exceedances
                )
            
            # Extract exceedances
            exceedances = extract_exceedances(window_returns, threshold)
            
            if len(exceedances) < min_exceedances:
                # Fallback: use empirical VaR
                var_series.iloc[i-1] = -window_returns.quantile(1 - confidence_level)
                continue
            
            # Fit GPD
            xi, beta, _ = fit_gpd_mle(exceedances, xi_lower, xi_upper)
            
            if np.isnan(xi) or np.isnan(beta) or beta <= 0:
                # Fallback: use empirical VaR
                var_series.iloc[i-1] = -window_returns.quantile(1 - confidence_level)
                continue
            
            # Compute VaR
            var_value = compute_var_from_evt(
                window_returns,
                threshold,
                xi,
                beta,
                confidence_level,
                horizon
            )
            
            var_series.iloc[i-1] = var_value
            
        except Exception as e:
            # Fallback: use empirical VaR
            var_series.iloc[i-1] = -window_returns.quantile(1 - confidence_level)
    
    return var_series


def compute_rolling_cvar_from_evt(
    returns: pd.Series,
    var_series: pd.Series,
    window: int = 252,
    confidence_level: float = 0.95,
    horizon: int = 1,
    threshold_quantiles: list = [0.90, 0.95, 0.97],
    min_exceedances: int = 50,
    xi_lower: float = -0.5,
    xi_upper: float = 0.5,
    automatic_threshold: bool = True
) -> pd.Series:
    """
    Compute rolling CVaR using EVT-POT methodology.
    
    Args:
        returns: Series of returns
        var_series: Series of VaR values (already computed)
        window: Rolling window size
        confidence_level: Confidence level
        horizon: Time horizon in days
        threshold_quantiles: List of quantiles to try for threshold
        min_exceedances: Minimum number of exceedances
        xi_lower: Lower bound for shape parameter
        xi_upper: Upper bound for shape parameter
        automatic_threshold: If True, automatically select threshold
        
    Returns:
        Series of CVaR values
    """
    cvar_series = pd.Series(index=returns.index, dtype=float)
    
    for i in range(window, len(returns) + 1):
        window_returns = returns.iloc[i-window:i]
        var_value = var_series.iloc[i-1]
        
        if np.isnan(var_value):
            cvar_series.iloc[i-1] = np.nan
            continue
        
        try:
            # Select threshold
            if automatic_threshold:
                threshold, num_exceedances, _ = select_threshold_automatic(
                    window_returns,
                    threshold_quantiles,
                    min_exceedances
                )
            else:
                threshold, num_exceedances = select_threshold_quantile(
                    window_returns,
                    threshold_quantiles[0] if threshold_quantiles else 0.95,
                    min_exceedances
                )
            
            # Extract exceedances
            exceedances = extract_exceedances(window_returns, threshold)
            
            if len(exceedances) < min_exceedances:
                # Fallback: use empirical CVaR
                losses = -window_returns
                p_target = 1 - confidence_level
                p_target_horizon = 1 - (confidence_level ** (1.0 / horizon))
                cvar_series.iloc[i-1] = losses[losses >= losses.quantile(1 - p_target_horizon)].mean()
                continue
            
            # Fit GPD
            xi, beta, _ = fit_gpd_mle(exceedances, xi_lower, xi_upper)
            
            if np.isnan(xi) or np.isnan(beta) or beta <= 0:
                # Fallback: use empirical CVaR
                losses = -window_returns
                p_target = 1 - confidence_level
                p_target_horizon = 1 - (confidence_level ** (1.0 / horizon))
                cvar_series.iloc[i-1] = losses[losses >= losses.quantile(1 - p_target_horizon)].mean()
                continue
            
            # Compute CVaR
            cvar_value = compute_cvar_from_evt(
                window_returns,
                threshold,
                xi,
                beta,
                var_value,
                confidence_level,
                horizon
            )
            
            cvar_series.iloc[i-1] = cvar_value
            
        except Exception as e:
            # Fallback: use empirical CVaR
            losses = -window_returns
            p_target = 1 - confidence_level
            p_target_horizon = 1 - (confidence_level ** (1.0 / horizon))
            cvar_series.iloc[i-1] = losses[losses >= losses.quantile(1 - p_target_horizon)].mean()
    
    return cvar_series


def align_returns_and_var(
    returns: pd.Series,
    var_series: pd.Series
) -> Tuple[pd.Series, pd.Series]:
    """
    Align returns and VaR series to common dates.
    
    Args:
        returns: Series of returns
        var_series: Series of VaR values
        
    Returns:
        Tuple of (aligned_returns, aligned_var)
    """
    common_dates = returns.index.intersection(var_series.index)
    aligned_returns = returns.loc[common_dates]
    aligned_var = var_series.loc[common_dates].dropna()
    
    # Further align to drop NaN VaR values
    aligned_returns = aligned_returns.loc[aligned_var.index]
    
    return aligned_returns, aligned_var


def align_returns_and_cvar(
    returns: pd.Series,
    cvar_series: pd.Series
) -> Tuple[pd.Series, pd.Series]:
    """
    Align returns and CVaR series to common dates.
    
    Args:
        returns: Series of returns
        cvar_series: Series of CVaR values
        
    Returns:
        Tuple of (aligned_returns, aligned_cvar)
    """
    common_dates = returns.index.intersection(cvar_series.index)
    aligned_returns = returns.loc[common_dates]
    aligned_cvar = cvar_series.loc[common_dates].dropna()
    
    # Further align to drop NaN CVaR values
    aligned_returns = aligned_returns.loc[aligned_cvar.index]
    
    return aligned_returns, aligned_cvar

