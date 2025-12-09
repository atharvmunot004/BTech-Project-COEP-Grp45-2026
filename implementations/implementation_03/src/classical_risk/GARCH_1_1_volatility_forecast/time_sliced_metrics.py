"""
Time-sliced metrics computation for GARCH VaR and CVaR backtesting.

Computes metrics for specific time periods (e.g., by year) to analyze
temporal patterns in VaR and CVaR performance.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from .backtesting import (
    detect_var_violations,
    detect_cvar_violations,
    compute_hit_rate,
    compute_violation_ratio
)


def compute_time_sliced_metrics(
    returns: pd.Series,
    var_series: pd.Series,
    cvar_series: Optional[pd.Series] = None,
    confidence_level: float = 0.95,
    slice_by: str = 'year'
) -> List[Dict]:
    """
    Compute metrics for time slices (e.g., by year).
    
    Args:
        returns: Series of actual returns with dates as index
        var_series: Series of VaR values with dates as index
        cvar_series: Optional Series of CVaR values with dates as index
        confidence_level: Confidence level used for VaR/CVaR
        slice_by: How to slice time ('year', 'quarter', 'month')
        
    Returns:
        List of dictionaries with metrics for each time slice
    """
    # Align returns and VaR
    common_dates = returns.index.intersection(var_series.index)
    if len(common_dates) == 0:
        return []
    
    aligned_returns = returns.loc[common_dates]
    aligned_var = var_series.loc[common_dates]
    
    # Align CVaR if provided
    aligned_cvar = None
    if cvar_series is not None:
        cvar_common_dates = returns.index.intersection(cvar_series.index)
        if len(cvar_common_dates) > 0:
            aligned_cvar = cvar_series.loc[cvar_common_dates]
            # Use intersection of both
            common_dates = common_dates.intersection(cvar_common_dates)
            aligned_returns = returns.loc[common_dates]
            aligned_var = var_series.loc[common_dates]
            aligned_cvar = cvar_series.loc[common_dates]
    
    # Create time slices
    if slice_by == 'year':
        time_groups = aligned_returns.index.year
        slice_format = lambda year: str(year)
    elif slice_by == 'quarter':
        time_groups = aligned_returns.index.to_period('Q')
        slice_format = lambda period: str(period)
    elif slice_by == 'month':
        time_groups = aligned_returns.index.to_period('M')
        slice_format = lambda period: str(period)
    else:
        raise ValueError(f"Unknown slice_by: {slice_by}. Use 'year', 'quarter', or 'month'")
    
    time_slices = []
    expected_violation_rate = 1 - confidence_level
    
    for time_period in sorted(time_groups.unique()):
        # Get data for this time period
        mask = time_groups == time_period
        period_returns = aligned_returns[mask]
        period_var = aligned_var[mask]
        period_cvar = aligned_cvar[mask] if aligned_cvar is not None else None
        
        if len(period_returns) == 0:
            continue
        
        # Compute VaR violations
        var_violations = detect_var_violations(period_returns, period_var, confidence_level)
        
        # Compute VaR metrics
        hit_rate = compute_hit_rate(var_violations)
        num_violations = var_violations.sum()
        expected_violations = len(var_violations) * expected_violation_rate
        violation_ratio_val = compute_violation_ratio(var_violations, expected_violation_rate)
        
        # Get date range
        period_dates = period_returns.index
        start_date = period_dates.min().strftime('%Y-%m-%d')
        end_date = period_dates.max().strftime('%Y-%m-%d')
        
        time_slice = {
            'slice': slice_format(time_period),
            'start_date': start_date,
            'end_date': end_date,
            'hit_rate': float(hit_rate),
            'num_violations': int(num_violations),
            'expected_violations': float(expected_violations),
            'violation_ratio': float(violation_ratio_val)
        }
        
        # Add CVaR metrics if available
        if period_cvar is not None:
            cvar_violations = detect_cvar_violations(
                period_returns, period_cvar, period_var, confidence_level
            )
            cvar_hit_rate = compute_hit_rate(cvar_violations)
            cvar_num_violations = cvar_violations.sum()
            time_slice['cvar_hit_rate'] = float(cvar_hit_rate)
            time_slice['cvar_num_violations'] = int(cvar_num_violations)
        
        time_slices.append(time_slice)
    
    return time_slices

