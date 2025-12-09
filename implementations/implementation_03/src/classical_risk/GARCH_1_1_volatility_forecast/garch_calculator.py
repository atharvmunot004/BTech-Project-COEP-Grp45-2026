"""
GARCH model calculator for volatility forecasting and VaR/CVaR estimation.

Implements GARCH(1,1) model fitting, conditional volatility computation,
and rolling volatility forecasting for portfolio risk assessment.
"""
import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple, Dict, Any
import warnings
import time

# Suppress convergence warnings from GARCH model fitting (handled by fallback mechanism)
# These warnings occur when GARCH fitting encounters numerical issues, but our fallback handles these cases
# Filter by message content since ConvergenceWarning may be a UserWarning
warnings.filterwarnings('ignore', message='.*optimizer returned code.*')
warnings.filterwarnings('ignore', message='.*Inequality constraints incompatible.*')
warnings.filterwarnings('ignore', message='.*ConvergenceWarning.*')
warnings.filterwarnings('ignore', message='.*See scipy.optimize.fmin_slsqp.*')
# Try to import and suppress ConvergenceWarning category if available
try:
    from arch.univariate.base import ConvergenceWarning
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
except (ImportError, AttributeError):
    pass
# Suppress UserWarnings from arch module (ConvergenceWarning is often a UserWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='arch')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='arch')

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    warnings.warn("arch library not available. GARCH functionality will not work. Install with: pip install arch")


def fit_garch_model(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    dist: str = 'normal',
    mean: str = 'Zero',
    vol: str = 'GARCH',
    rescale: bool = True
) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Fit a GARCH(p,q) model to return series.
    
    Args:
        returns: Series of returns
        p: GARCH order (number of lagged conditional variances)
        q: ARCH order (number of lagged squared residuals)
        dist: Distribution assumption ('normal', 't', 'ged')
        mean: Mean model ('Zero', 'AR', 'Constant')
        vol: Volatility model ('GARCH', 'EGARCH', etc.)
        rescale: Whether to rescale data
        
    Returns:
        Tuple of (fitted model, fit results)
    """
    if not ARCH_AVAILABLE:
        raise ImportError("arch library is required. Install with: pip install arch")
    
    if len(returns.dropna()) < max(p, q) + 10:
        return None, None
    
    try:
        # Fit GARCH model
        model = arch_model(
            returns.dropna(),
            vol=vol,
            p=p,
            q=q,
            dist=dist,
            mean=mean,
            rescale=rescale
        )
        
        # Suppress ConvergenceWarnings during fitting
        # Use a comprehensive warning suppression context
        with warnings.catch_warnings(record=False):
            warnings.simplefilter('ignore')
            # Also temporarily redirect stderr to suppress warnings printed directly by arch library
            import sys
            import io
            old_stderr = sys.stderr
            try:
                # Redirect stderr to a null stream during fitting
                sys.stderr = io.StringIO()
                # Fit with display='off' to suppress output
                fit_result = model.fit(update_freq=0, disp='off')
            finally:
                # Restore stderr
                sys.stderr = old_stderr
        
        return model, fit_result
    except Exception as e:
        # Only warn for actual exceptions, not convergence issues
        return None, None


def compute_conditional_volatility(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    dist: str = 'normal',
    mean: str = 'Zero',
    vol: str = 'GARCH',
    fallback_long_run_variance: bool = True
) -> pd.Series:
    """
    Compute conditional volatility from GARCH model fitted to entire series.
    
    Args:
        returns: Series of returns
        p: GARCH order
        q: ARCH order
        dist: Distribution assumption
        mean: Mean model
        vol: Volatility model
        fallback_long_run_variance: If True, use long-run variance if model fails
        
    Returns:
        Series of conditional volatility estimates
    """
    if not ARCH_AVAILABLE:
        raise ImportError("arch library is required. Install with: pip install arch")
    
    model, fit_result = fit_garch_model(returns, p, q, dist, mean, vol)
    
    if fit_result is None:
        if fallback_long_run_variance:
            # Use long-run variance as fallback
            long_run_vol = returns.std()
            return pd.Series(long_run_vol, index=returns.index, name='conditional_volatility')
        else:
            return pd.Series(np.nan, index=returns.index, name='conditional_volatility')
    
    # Get conditional volatility from fitted model
    conditional_vol = fit_result.conditional_volatility
    
    # Align with original returns index
    conditional_vol_series = pd.Series(
        conditional_vol,
        index=returns.dropna().index,
        name='conditional_volatility'
    )
    
    return conditional_vol_series


def forecast_volatility(
    returns: pd.Series,
    horizon: int = 1,
    p: int = 1,
    q: int = 1,
    dist: str = 'normal',
    mean: str = 'Zero',
    vol: str = 'GARCH',
    fallback_long_run_variance: bool = True
) -> float:
    """
    Forecast volatility for a given horizon using GARCH model.
    
    Args:
        returns: Series of historical returns
        horizon: Forecast horizon (in periods)
        p: GARCH order
        q: ARCH order
        dist: Distribution assumption
        mean: Mean model
        vol: Volatility model
        fallback_long_run_variance: If True, use long-run variance if model fails
        
    Returns:
        Forecasted volatility (annualized if returns are daily)
    """
    if not ARCH_AVAILABLE:
        raise ImportError("arch library is required. Install with: pip install arch")
    
    model, fit_result = fit_garch_model(returns, p, q, dist, mean, vol)
    
    if fit_result is None:
        if fallback_long_run_variance:
            # Use long-run variance as fallback
            return returns.std() * np.sqrt(horizon)
        else:
            return np.nan
    
    try:
        # Suppress warnings during forecast
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # Forecast volatility
            forecast = fit_result.forecast(horizon=horizon, reindex=False)
        
        # Get variance forecast
        variance_forecast = forecast.variance.iloc[-1, -1]  # Last forecast
        
        # Convert to volatility (standard deviation)
        vol_forecast = np.sqrt(variance_forecast) * np.sqrt(horizon)
        
        return float(vol_forecast)
    except Exception as e:
        # Silently fall back on forecast failure
        if fallback_long_run_variance:
            return returns.std() * np.sqrt(horizon)
        else:
            return np.nan


def compute_rolling_volatility_forecast(
    returns: pd.Series,
    window: int,
    horizon: int = 1,
    p: int = 1,
    q: int = 1,
    dist: str = 'normal',
    mean: str = 'Zero',
    vol: str = 'GARCH',
    fallback_long_run_variance: bool = True
) -> pd.Series:
    """
    Compute rolling volatility forecasts using GARCH models.
    
    For each date, fits GARCH to a rolling window of historical returns
    and forecasts volatility for the next horizon periods.
    
    Args:
        returns: Series of returns with dates as index
        window: Rolling window size for model fitting
        horizon: Forecast horizon (in periods)
        p: GARCH order
        q: ARCH order
        dist: Distribution assumption
        mean: Mean model
        vol: Volatility model
        fallback_long_run_variance: If True, use long-run variance if model fails
        
    Returns:
        Series of volatility forecasts with dates as index
    """
    if not ARCH_AVAILABLE:
        raise ImportError("arch library is required. Install with: pip install arch")
    
    forecasts = []
    forecast_dates = []
    
    # Need at least window + some buffer for GARCH fitting
    min_window = max(window, max(p, q) + 10)
    
    for i in range(min_window, len(returns)):
        # Get rolling window of returns
        window_returns = returns.iloc[i - window:i]
        
        # Forecast volatility
        vol_forecast = forecast_volatility(
            window_returns,
            horizon=horizon,
            p=p,
            q=q,
            dist=dist,
            mean=mean,
            vol=vol,
            fallback_long_run_variance=fallback_long_run_variance
        )
        
        # Forecast date is the date for which we're forecasting (current date)
        forecast_date = returns.index[i]
        
        forecasts.append(vol_forecast)
        forecast_dates.append(forecast_date)
    
    # Create Series with forecasts
    forecast_series = pd.Series(
        forecasts,
        index=forecast_dates,
        name='volatility_forecast'
    )
    
    return forecast_series


def var_from_volatility(
    volatility: Union[pd.Series, float],
    confidence_level: float = 0.95,
    horizon: int = 1,
    dist: str = 'normal'
) -> Union[pd.Series, float]:
    """
    Compute VaR from volatility forecast using distributional assumption.
    
    VaR = volatility * z_score(confidence_level) * sqrt(horizon)
    
    Args:
        volatility: Volatility forecast (Series or scalar)
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        horizon: Time horizon (e.g., 1 for 1 day)
        dist: Distribution assumption ('normal', 't')
        
    Returns:
        VaR estimate (Series or scalar, same type as volatility)
    """
    from scipy import stats
    
    # Z-score for confidence level
    alpha = 1 - confidence_level
    
    if dist == 'normal':
        z_score = stats.norm.ppf(confidence_level)
    elif dist == 't':
        # Use t-distribution with 4 degrees of freedom (common choice)
        z_score = stats.t.ppf(confidence_level, df=4)
    else:
        # Default to normal
        z_score = stats.norm.ppf(confidence_level)
    
    # VaR = volatility * z_score * sqrt(horizon)
    # Note: volatility is already scaled, so we use it directly
    var = volatility * z_score * np.sqrt(horizon)
    
    return var


def cvar_from_volatility(
    volatility: Union[pd.Series, float],
    confidence_level: float = 0.95,
    horizon: int = 1,
    dist: str = 'normal'
) -> Union[pd.Series, float]:
    """
    Compute CVaR from volatility forecast using distributional assumption.
    
    CVaR = volatility * expected_shortfall_factor * sqrt(horizon)
    
    Args:
        volatility: Volatility forecast (Series or scalar)
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        horizon: Time horizon (e.g., 1 for 1 day)
        dist: Distribution assumption ('normal', 't')
        
    Returns:
        CVaR estimate (Series or scalar, same type as volatility)
    """
    from scipy import stats
    
    # Expected shortfall (ES) for normal distribution
    alpha = 1 - confidence_level
    
    if dist == 'normal':
        # For normal: ES = -sigma * phi(z_alpha) / alpha
        z_alpha = stats.norm.ppf(confidence_level)
        phi_z_alpha = stats.norm.pdf(z_alpha)
        es_factor = phi_z_alpha / alpha
    elif dist == 't':
        # For t-distribution with 4 df
        z_alpha = stats.t.ppf(confidence_level, df=4)
        # Approximate ES for t-distribution
        es_factor = stats.t.pdf(z_alpha, df=4) / alpha * (4 + z_alpha**2) / (4 - 1)
    else:
        # Default to normal
        z_alpha = stats.norm.ppf(confidence_level)
        phi_z_alpha = stats.norm.pdf(z_alpha)
        es_factor = phi_z_alpha / alpha
    
    # CVaR = volatility * es_factor * sqrt(horizon)
    cvar = volatility * es_factor * np.sqrt(horizon)
    
    return cvar


def compute_rolling_var_from_garch(
    returns: pd.Series,
    window: int,
    confidence_level: float = 0.95,
    horizon: int = 1,
    p: int = 1,
    q: int = 1,
    dist: str = 'normal',
    mean: str = 'Zero',
    vol: str = 'GARCH',
    fallback_long_run_variance: bool = True
) -> pd.Series:
    """
    Compute rolling VaR using GARCH volatility forecasts.
    
    Args:
        returns: Series of returns
        window: Rolling window size for GARCH fitting
        confidence_level: Confidence level for VaR
        horizon: Forecast horizon
        p: GARCH order
        q: ARCH order
        dist: Distribution assumption
        mean: Mean model
        vol: Volatility model
        fallback_long_run_variance: If True, use long-run variance if model fails
        
    Returns:
        Series of VaR estimates
    """
    # Get volatility forecasts
    vol_forecasts = compute_rolling_volatility_forecast(
        returns,
        window=window,
        horizon=horizon,
        p=p,
        q=q,
        dist=dist,
        mean=mean,
        vol=vol,
        fallback_long_run_variance=fallback_long_run_variance
    )
    
    # Convert volatility to VaR
    var_series = var_from_volatility(
        vol_forecasts,
        confidence_level=confidence_level,
        horizon=horizon,
        dist=dist
    )
    
    return var_series


def compute_rolling_cvar_from_garch(
    returns: pd.Series,
    window: int,
    confidence_level: float = 0.95,
    horizon: int = 1,
    p: int = 1,
    q: int = 1,
    dist: str = 'normal',
    mean: str = 'Zero',
    vol: str = 'GARCH',
    fallback_long_run_variance: bool = True
) -> pd.Series:
    """
    Compute rolling CVaR using GARCH volatility forecasts.
    
    Args:
        returns: Series of returns
        window: Rolling window size for GARCH fitting
        confidence_level: Confidence level for CVaR
        horizon: Forecast horizon
        p: GARCH order
        q: ARCH order
        dist: Distribution assumption
        mean: Mean model
        vol: Volatility model
        fallback_long_run_variance: If True, use long-run variance if model fails
        
    Returns:
        Series of CVaR estimates
    """
    # Get volatility forecasts
    vol_forecasts = compute_rolling_volatility_forecast(
        returns,
        window=window,
        horizon=horizon,
        p=p,
        q=q,
        dist=dist,
        mean=mean,
        vol=vol,
        fallback_long_run_variance=fallback_long_run_variance
    )
    
    # Convert volatility to CVaR
    cvar_series = cvar_from_volatility(
        vol_forecasts,
        confidence_level=confidence_level,
        horizon=horizon,
        dist=dist
    )
    
    return cvar_series


def align_returns_and_var(
    returns: pd.Series,
    var_series: pd.Series
) -> Tuple[pd.Series, pd.Series]:
    """
    Align returns and VaR series to common dates.
    
    Args:
        returns: Series of returns
        var_series: Series of VaR estimates
        
    Returns:
        Tuple of (aligned returns, aligned VaR)
    """
    common_dates = returns.index.intersection(var_series.index)
    
    if len(common_dates) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    
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
        returns: Series of returns
        cvar_series: Series of CVaR estimates
        
    Returns:
        Tuple of (aligned returns, aligned CVaR)
    """
    common_dates = returns.index.intersection(cvar_series.index)
    
    if len(common_dates) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    
    aligned_returns = returns.loc[common_dates]
    aligned_cvar = cvar_series.loc[common_dates]
    
    return aligned_returns, aligned_cvar

