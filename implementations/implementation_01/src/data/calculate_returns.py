"""
Calculate daily returns (simple and log returns) from price data.
"""
import pandas as pd
import numpy as np
from typing import Literal


def calculate_returns(
    prices: pd.DataFrame,
    method: Literal["simple", "log"] = "log",
    fill_na: bool = True
) -> pd.DataFrame:
    """
    Calculate returns from price data.
    
    Args:
        prices: DataFrame with dates as index and symbols as columns (or single Series)
        method: "simple" for simple returns, "log" for log returns
        fill_na: Whether to forward-fill NaN values
        
    Returns:
        DataFrame/Series of returns with same structure as input
    """
    if method == "simple":
        returns = prices.pct_change()
    elif method == "log":
        returns = np.log(prices / prices.shift(1))
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if fill_na:
        returns = returns.ffill().fillna(0)
    
    return returns


def calculate_volatilities(
    returns: pd.DataFrame,
    window: int = 252,
    annualize: bool = True
) -> pd.Series:
    """
    Calculate rolling volatility (standard deviation of returns).
    
    Args:
        returns: DataFrame of returns
        window: Rolling window size (default 252 for annual)
        annualize: Whether to annualize (multiply by sqrt(252))
        
    Returns:
        Series of volatilities
    """
    vol = returns.std() * np.sqrt(window) if annualize else returns.std()
    return vol


if __name__ == "__main__":
    # Test
    from data.load_data import DataLoader
    
    loader = DataLoader()
    prices = loader.load_multiple_stocks(["RELIANCE", "TCS"])
    returns = calculate_returns(prices, method="log")
    print("Returns shape:", returns.shape)
    print(returns.head())
    
    volatilities = calculate_volatilities(returns)
    print("\nVolatilities:")
    print(volatilities)

