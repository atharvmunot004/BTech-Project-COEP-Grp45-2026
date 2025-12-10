"""
Returns computation module for Markowitz optimization.

Computes daily returns from price data and loads portfolio universe.
"""
import pandas as pd
import numpy as np
from typing import Union, Optional
from pathlib import Path


def compute_daily_returns(
    prices: pd.DataFrame,
    method: str = 'log'
) -> pd.DataFrame:
    """
    Compute daily returns from price data.
    
    Args:
        prices: DataFrame with dates as index and assets as columns
        method: 'log' for log returns, 'simple' for simple returns
        
    Returns:
        DataFrame of daily returns with same index and columns as prices
    """
    if method == 'log':
        returns = np.log(prices / prices.shift(1))
    elif method == 'simple':
        returns = (prices / prices.shift(1)) - 1
    else:
        raise ValueError(f"Unknown method: {method}. Use 'log' or 'simple'")
    
    # Drop first row (NaN)
    returns = returns.dropna()
    
    return returns


def load_panel_prices(panel_price_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load panel price data from parquet or CSV file.
    
    Args:
        panel_price_path: Path to panel price file
        
    Returns:
        DataFrame with dates as index and assets as columns
    """
    # Store original path string for relative resolution
    original_path = str(panel_price_path)
    panel_price_path = Path(panel_price_path)
    
    # If path is absolute and exists, use it directly
    if panel_price_path.is_absolute() and panel_price_path.exists():
        pass  # Use as-is
    elif panel_price_path.exists():
        pass  # Relative path that exists
    else:
        # Try relative to implementation_03 root
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent
        # Use original path string to ensure proper relative resolution
        panel_price_path = project_root / original_path
        
        if not panel_price_path.exists():
            # Collect all attempted paths for better error message
            attempted_paths = []
            # First attempt: original path as-is
            first_attempt = Path(original_path)
            if first_attempt.is_absolute():
                attempted_paths.append(str(first_attempt))
            else:
                attempted_paths.append(str(first_attempt.resolve()))
            # Second attempt: relative to project root
            attempted_paths.append(str(panel_price_path))
            
            raise FileNotFoundError(
                f"Panel price file not found. Attempted paths:\n"
                f"  1. {attempted_paths[0]}\n"
                f"  2. {attempted_paths[1]}\n"
                f"Please check that the file exists or update the path in llm.json"
            )
    
    if panel_price_path.suffix == '.parquet':
        prices = pd.read_parquet(panel_price_path)
    elif panel_price_path.suffix == '.csv':
        prices = pd.read_csv(panel_price_path, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"Unsupported file format: {panel_price_path.suffix}")
    
    # Ensure index is datetime
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.to_datetime(prices.index)
    
    # Sort by date
    prices = prices.sort_index()
    
    return prices


def load_portfolio_universe(portfolio_universe_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load portfolio universe from parquet or CSV file.
    
    Args:
        portfolio_universe_path: Path to portfolio universe file
        
    Returns:
        DataFrame with portfolio identifiers and asset information
    """
    # Store original path string for relative resolution
    original_path = str(portfolio_universe_path)
    portfolio_universe_path = Path(portfolio_universe_path)
    
    # If path is absolute and exists, use it directly
    if portfolio_universe_path.is_absolute() and portfolio_universe_path.exists():
        pass  # Use as-is
    elif portfolio_universe_path.exists():
        pass  # Relative path that exists
    else:
        # Try relative to implementation_03 root
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent
        # Use original path string to ensure proper relative resolution
        portfolio_universe_path = project_root / original_path
        
        if not portfolio_universe_path.exists():
            attempted_paths = [
                str(Path(original_path).resolve()),
                str(portfolio_universe_path)
            ]
            raise FileNotFoundError(
                f"Portfolio universe file not found. Attempted paths:\n"
                f"  1. {attempted_paths[0]}\n"
                f"  2. {attempted_paths[1]}\n"
                f"Please check that the file exists or update the path in llm.json"
            )
    
    if portfolio_universe_path.suffix == '.parquet':
        universe = pd.read_parquet(portfolio_universe_path)
    elif portfolio_universe_path.suffix == '.csv':
        universe = pd.read_csv(portfolio_universe_path, index_col=0)
    else:
        raise ValueError(f"Unsupported file format: {portfolio_universe_path.suffix}")
    
    return universe

