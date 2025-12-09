"""
Returns computation module for VaR evaluation.

Computes daily returns from price data and portfolio returns from portfolio weights.
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


def compute_portfolio_returns(
    asset_returns: pd.DataFrame,
    portfolio_weights: pd.Series,
    align_assets: bool = True
) -> pd.Series:
    """
    Compute portfolio returns from asset returns and portfolio weights.
    
    Args:
        asset_returns: DataFrame with dates as index and assets as columns
        portfolio_weights: Series with assets as index and weights as values
        align_assets: If True, align assets between returns and weights
        
    Returns:
        Series of portfolio returns with dates as index
    """
    if align_assets:
        # Get common assets
        common_assets = asset_returns.columns.intersection(portfolio_weights.index)
        if len(common_assets) == 0:
            raise ValueError("No common assets between returns and weights")
        
        asset_returns = asset_returns[common_assets]
        portfolio_weights = portfolio_weights[common_assets]
    
    # Normalize weights to sum to 1
    portfolio_weights = portfolio_weights / portfolio_weights.sum()
    
    # Compute portfolio returns: weighted sum of asset returns
    portfolio_returns = (asset_returns * portfolio_weights).sum(axis=1)
    
    return portfolio_returns


def load_panel_prices(panel_price_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load panel price data from parquet or CSV file.
    
    Args:
        panel_price_path: Path to panel price file
        
    Returns:
        DataFrame with dates as index and assets as columns
    """
    panel_price_path = Path(panel_price_path)
    
    if not panel_price_path.exists():
        # Try relative to implementation_03 root
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent
        panel_price_path = project_root / panel_price_path
        
        if not panel_price_path.exists():
            raise FileNotFoundError(f"Panel price file not found: {panel_price_path}")
    
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


def load_portfolio_weights(portfolio_weights_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load portfolio weights from parquet or CSV file.
    
    Args:
        portfolio_weights_path: Path to portfolio weights file
        
    Returns:
        DataFrame with portfolios as rows and assets as columns
    """
    portfolio_weights_path = Path(portfolio_weights_path)
    
    if not portfolio_weights_path.exists():
        # Try relative to implementation_03 root
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent
        portfolio_weights_path = project_root / portfolio_weights_path
        
        if not portfolio_weights_path.exists():
            raise FileNotFoundError(f"Portfolio weights file not found: {portfolio_weights_path}")
    
    if portfolio_weights_path.suffix == '.parquet':
        weights = pd.read_parquet(portfolio_weights_path)
    elif portfolio_weights_path.suffix == '.csv':
        weights = pd.read_csv(portfolio_weights_path, index_col=0)
    else:
        raise ValueError(f"Unsupported file format: {portfolio_weights_path.suffix}")
    
    return weights

