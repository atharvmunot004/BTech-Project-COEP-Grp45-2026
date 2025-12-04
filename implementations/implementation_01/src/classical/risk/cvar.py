"""
Conditional Value-at-Risk (CVaR) / Expected Shortfall calculation.
"""
import numpy as np
import pandas as pd
from typing import Union, Optional
from scipy import stats


def calculate_cvar(
    returns: Union[pd.Series, pd.DataFrame, np.ndarray],
    portfolio_weights: Optional[np.ndarray] = None,
    confidence_level: float = 0.95,
    method: str = "analytical"
) -> float:
    """
    Calculate Conditional Value-at-Risk (CVaR) / Expected Shortfall.
    
    CVaR is the expected loss given that the loss exceeds VaR.
    
    Args:
        returns: Returns data
        portfolio_weights: Portfolio weights
        confidence_level: Confidence level
        method: "analytical" (assumes normal) or "empirical"
        
    Returns:
        CVaR value
    """
    if method == "analytical":
        return _analytical_cvar(returns, portfolio_weights, confidence_level)
    elif method == "empirical":
        return _empirical_cvar(returns, portfolio_weights, confidence_level)
    else:
        raise ValueError(f"Unknown method: {method}")


def _analytical_cvar(
    returns: Union[pd.Series, pd.DataFrame, np.ndarray],
    portfolio_weights: Optional[np.ndarray],
    confidence_level: float
) -> float:
    """Analytical CVaR assuming normal distribution."""
    # Convert to numpy
    if isinstance(returns, pd.DataFrame):
        returns_array = returns.values
    elif isinstance(returns, pd.Series):
        returns_array = returns.values.reshape(-1, 1)
    else:
        returns_array = np.array(returns)
        if returns_array.ndim == 1:
            returns_array = returns_array.reshape(-1, 1)
    
    # Portfolio weights
    if portfolio_weights is None:
        portfolio_weights = np.ones(returns_array.shape[1]) / returns_array.shape[1]
    else:
        portfolio_weights = np.array(portfolio_weights)
    
    # Calculate portfolio returns
    portfolio_returns = np.dot(returns_array, portfolio_weights)
    
    # Mean and std
    mean_return = np.mean(portfolio_returns)
    std_return = np.std(portfolio_returns, ddof=1)
    
    # Annualize
    mean_return = mean_return * 252
    std_return = std_return * np.sqrt(252)
    
    # VaR
    alpha = 1 - confidence_level
    z_alpha = stats.norm.ppf(alpha)
    var = -mean_return - z_alpha * std_return
    
    # CVaR for normal distribution
    # CVaR = -μ - σ * φ(z_α) / α
    phi_z_alpha = stats.norm.pdf(z_alpha)
    cvar = -mean_return - std_return * phi_z_alpha / alpha
    
    return cvar


def _empirical_cvar(
    returns: Union[pd.Series, pd.DataFrame, np.ndarray],
    portfolio_weights: Optional[np.ndarray],
    confidence_level: float
) -> float:
    """Empirical CVaR from historical data."""
    # Convert to numpy
    if isinstance(returns, pd.DataFrame):
        returns_array = returns.values
    elif isinstance(returns, pd.Series):
        returns_array = returns.values.reshape(-1, 1)
    else:
        returns_array = np.array(returns)
        if returns_array.ndim == 1:
            returns_array = returns_array.reshape(-1, 1)
    
    # Portfolio weights
    if portfolio_weights is None:
        portfolio_weights = np.ones(returns_array.shape[1]) / returns_array.shape[1]
    else:
        portfolio_weights = np.array(portfolio_weights)
    
    # Portfolio returns
    portfolio_returns = np.dot(returns_array, portfolio_weights)
    
    # VaR threshold
    alpha = 1 - confidence_level
    var_threshold = -np.percentile(portfolio_returns, alpha * 100)
    
    # CVaR: mean of losses exceeding VaR
    tail_losses = portfolio_returns[portfolio_returns <= -var_threshold]
    if len(tail_losses) > 0:
        cvar = -np.mean(tail_losses)
    else:
        cvar = var_threshold
    
    return cvar


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
    
    cvar_analytical = calculate_cvar(returns, weights, method="analytical")
    cvar_empirical = calculate_cvar(returns, weights, method="empirical")
    
    print(f"Analytical CVaR (95%): {cvar_analytical:.4f}")
    print(f"Empirical CVaR (95%): {cvar_empirical:.4f}")

