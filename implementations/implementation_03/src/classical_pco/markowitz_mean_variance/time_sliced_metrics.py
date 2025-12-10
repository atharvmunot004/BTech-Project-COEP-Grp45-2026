"""
Time-sliced metrics computation module for Markowitz optimization.

Computes metrics sliced by time periods (e.g., by year).
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime


def compute_time_sliced_metrics(
    portfolio_returns: pd.Series,
    portfolio_weights: pd.Series,
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    slice_by: str = 'year',
    risk_free_rate: float = 0.0
) -> List[Dict]:
    """
    Compute portfolio metrics sliced by time periods.
    
    Args:
        portfolio_returns: Series of portfolio returns with dates as index
        portfolio_weights: Series of portfolio weights
        expected_returns: Series of expected returns
        covariance_matrix: Covariance matrix DataFrame
        slice_by: 'year', 'quarter', or 'month'
        risk_free_rate: Risk-free rate
        
    Returns:
        List of dictionaries with metrics for each time slice
    """
    if len(portfolio_returns) == 0:
        return []
    
    # Ensure index is datetime
    if not isinstance(portfolio_returns.index, pd.DatetimeIndex):
        portfolio_returns.index = pd.to_datetime(portfolio_returns.index)
    
    # Group by time period
    if slice_by == 'year':
        groups = portfolio_returns.groupby(portfolio_returns.index.year)
    elif slice_by == 'quarter':
        groups = portfolio_returns.groupby([
            portfolio_returns.index.year,
            portfolio_returns.index.quarter
        ])
    elif slice_by == 'month':
        groups = portfolio_returns.groupby([
            portfolio_returns.index.year,
            portfolio_returns.index.month
        ])
    else:
        raise ValueError(f"Unknown slice_by: {slice_by}. Use 'year', 'quarter', or 'month'")
    
    time_slices = []
    
    for period, period_returns in groups:
        if len(period_returns) < 2:
            continue
        
        # Compute metrics for this period
        period_metrics = {
            'period': str(period) if isinstance(period, tuple) else str(period),
            'start_date': period_returns.index[0].strftime('%Y-%m-%d'),
            'end_date': period_returns.index[-1].strftime('%Y-%m-%d'),
            'num_observations': len(period_returns),
            'mean_return': float(period_returns.mean()),
            'volatility': float(period_returns.std()),
            'sharpe_ratio': float((period_returns.mean() - risk_free_rate) / period_returns.std() if period_returns.std() > 0 else np.nan),
            'min_return': float(period_returns.min()),
            'max_return': float(period_returns.max())
        }
        
        # Add cumulative return
        cumulative_return = (1 + period_returns).prod() - 1
        period_metrics['cumulative_return'] = float(cumulative_return)
        
        time_slices.append(period_metrics)
    
    return time_slices

