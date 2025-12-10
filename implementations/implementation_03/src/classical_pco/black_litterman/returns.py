"""
Returns computation and data loading module for Black-Litterman optimization.

Handles loading of prices and baseline portfolios.
"""
import pandas as pd
import numpy as np
from typing import Union
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
    original_path = str(panel_price_path)
    panel_price_path = Path(panel_price_path)
    
    if panel_price_path.is_absolute() and panel_price_path.exists():
        pass
    elif panel_price_path.exists():
        pass
    else:
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent
        panel_price_path = project_root / original_path
        
        if not panel_price_path.exists():
            attempted_paths = []
            first_attempt = Path(original_path)
            if first_attempt.is_absolute():
                attempted_paths.append(str(first_attempt))
            else:
                attempted_paths.append(str(first_attempt.resolve()))
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
    
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.to_datetime(prices.index)
    
    prices = prices.sort_index()
    
    return prices


def load_baseline_portfolios(baseline_portfolios_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load baseline portfolios from parquet or CSV file.
    
    Args:
        baseline_portfolios_path: Path to baseline portfolios file
        
    Returns:
        DataFrame with portfolio identifiers and asset information
    """
    original_path = str(baseline_portfolios_path)
    baseline_portfolios_path = Path(baseline_portfolios_path)
    
    if baseline_portfolios_path.is_absolute() and baseline_portfolios_path.exists():
        pass
    elif baseline_portfolios_path.exists():
        pass
    else:
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent
        baseline_portfolios_path = project_root / original_path
        
        if not baseline_portfolios_path.exists():
            attempted_paths = [
                str(Path(original_path).resolve()),
                str(baseline_portfolios_path)
            ]
            raise FileNotFoundError(
                f"Baseline portfolios file not found. Attempted paths:\n"
                f"  1. {attempted_paths[0]}\n"
                f"  2. {attempted_paths[1]}\n"
                f"Please check that the file exists or update the path in llm.json"
            )
    
    if baseline_portfolios_path.suffix == '.parquet':
        portfolios = pd.read_parquet(baseline_portfolios_path)
    elif baseline_portfolios_path.suffix == '.csv':
        portfolios = pd.read_csv(baseline_portfolios_path, index_col=0)
    else:
        raise ValueError(f"Unsupported file format: {baseline_portfolios_path.suffix}")
    
    return portfolios
