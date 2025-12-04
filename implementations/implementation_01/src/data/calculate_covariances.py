"""
Calculate covariance and correlation matrices from returns.
"""
import pandas as pd
import numpy as np
from typing import Optional


def calculate_covariance_matrix(
    returns: pd.DataFrame,
    method: str = "sample",
    annualize: bool = True
) -> pd.DataFrame:
    """
    Calculate covariance matrix from returns.
    
    Args:
        returns: DataFrame of returns (date x symbol)
        method: "sample" for sample covariance, "exponential" for EWMA
        annualize: Whether to annualize (multiply by 252)
        
    Returns:
        Covariance matrix (symbol x symbol)
    """
    if method == "sample":
        cov = returns.cov()
    elif method == "exponential":
        # Exponentially weighted moving average covariance
        cov = returns.ewm(span=60).cov().iloc[-len(returns.columns):]
        # Extract the last covariance matrix
        cov = cov.groupby(level=1).last()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if annualize:
        cov = cov * 252
    
    return cov


def calculate_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlation matrix from returns.
    
    Args:
        returns: DataFrame of returns (date x symbol)
        
    Returns:
        Correlation matrix (symbol x symbol)
    """
    return returns.corr()


def calculate_mean_returns(returns: pd.DataFrame, annualize: bool = True) -> pd.Series:
    """
    Calculate mean returns (expected returns).
    
    Args:
        returns: DataFrame of returns
        annualize: Whether to annualize (multiply by 252)
        
    Returns:
        Series of mean returns per symbol
    """
    mean_returns = returns.mean()
    if annualize:
        mean_returns = mean_returns * 252
    return mean_returns


if __name__ == "__main__":
    # Test
    from data.load_data import DataLoader
    from data.calculate_returns import calculate_returns
    
    loader = DataLoader()
    prices = loader.load_multiple_stocks(["RELIANCE", "TCS", "HDFCBANK"])
    returns = calculate_returns(prices)
    
    cov = calculate_covariance_matrix(returns)
    print("Covariance matrix:")
    print(cov)
    
    corr = calculate_correlation_matrix(returns)
    print("\nCorrelation matrix:")
    print(corr)
    
    mu = calculate_mean_returns(returns)
    print("\nMean returns:")
    print(mu)

