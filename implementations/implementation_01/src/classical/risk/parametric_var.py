"""
Variance-Covariance (Parametric) Value-at-Risk calculation.
"""
import numpy as np
import pandas as pd
from typing import Union, Optional
from scipy import stats


def parametric_var(
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
    
    # Per-asset VaR
    var_per_asset = parametric_var(returns, confidence_level=0.95, horizon=1)
    print("Per-asset VaR (95%, 1 day):")
    print(var_per_asset)
    
    # Portfolio VaR (equal weights)
    weights = np.array([1/3, 1/3, 1/3])
    portfolio_var = parametric_var(returns, portfolio_weights=weights, confidence_level=0.95, horizon=1)
    print(f"\nPortfolio VaR (equal weights, 95%, 1 day): {portfolio_var:.4f}")

