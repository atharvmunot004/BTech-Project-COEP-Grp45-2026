"""Rolling window backtesting."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
import time
from config.config_loader import get_config


class RollingWindowBacktester:
    """Rolling window backtesting framework."""
    
    def __init__(self):
        """Initialize the backtester."""
        self.config = get_config()
        backtest_config = self.config.get_backtesting_config()
        self.train_window_years = backtest_config.get('train_window_years', 3)
        self.test_horizon_years = backtest_config.get('test_horizon_years', 7)
        self.min_periods_days = backtest_config.get('min_periods_for_estimation_days', 252)
        self.rolling_step_days = backtest_config.get('rolling_step_days', 1)
        self.rebalance_frequency_days = backtest_config.get('evaluation_horizons', {}).get('portfolio_rebalance_frequency_days', 21)
    
    def create_rolling_windows(
        self,
        returns_df: pd.DataFrame
    ) -> List[Dict]:
        """
        Create rolling train/test windows.
        
        Args:
            returns_df: Returns DataFrame with Date index
            
        Returns:
            List of dictionaries with train/test date ranges
        """
        windows = []
        dates = returns_df.index.sort_values()
        
        train_days = int(self.train_window_years * 252)
        test_days = int(self.test_horizon_years * 252)
        
        # Ensure we have minimum data
        min_required_days = train_days + max(test_days, self.min_periods_days)
        
        if len(dates) < min_required_days:
            # If not enough data, use what we have: use 70% for train, 30% for test
            train_days = int(len(dates) * 0.7)
            test_days = len(dates) - train_days
            print(f"Warning: Not enough data for full windows. Using {train_days} train days and {test_days} test days.")
        
        start_idx = 0
        max_windows = 10  # Limit to prevent too many windows
        
        while start_idx + train_days < len(dates) and len(windows) < max_windows:
            train_start = dates[start_idx]
            train_end = dates[min(start_idx + train_days - 1, len(dates) - 1)]
            
            # Test period starts after training
            test_start_idx = start_idx + train_days
            if test_start_idx >= len(dates):
                break
            
            test_start = dates[test_start_idx]
            # Use remaining data or specified test horizon, whichever is smaller
            test_end_idx = min(start_idx + train_days + test_days, len(dates) - 1)
            test_end = dates[test_end_idx]
            
            # Only add window if test period has at least some data
            if test_end_idx > test_start_idx:
                windows.append({
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'window_id': len(windows)
                })
            
            # Move window
            start_idx += self.rolling_step_days
            
            # If we can't create more windows, break
            if start_idx + train_days >= len(dates):
                break
        
        # If no windows created, create at least one using available data
        if len(windows) == 0 and len(dates) >= self.min_periods_days:
            # Use first 70% for train, rest for test
            split_idx = int(len(dates) * 0.7)
            windows.append({
                'train_start': dates[0],
                'train_end': dates[split_idx - 1],
                'test_start': dates[split_idx],
                'test_end': dates[-1],
                'window_id': 0
            })
        
        return windows
    
    def backtest_method(
        self,
        returns_df: pd.DataFrame,
        method_func: Callable,
        method_name: str,
        windows: Optional[List[Dict]] = None
    ) -> pd.DataFrame:
        """
        Backtest a method across rolling windows.
        
        Args:
            returns_df: Returns DataFrame
            method_func: Function that takes (train_returns, test_returns) and returns results
            method_name: Name of the method
            windows: List of windows (if None, creates them)
            
        Returns:
            DataFrame with results per window
        """
        if windows is None:
            windows = self.create_rolling_windows(returns_df)
        
        results = []
        
        for window in windows:
            train_returns = returns_df.loc[window['train_start']:window['train_end']]
            test_returns = returns_df.loc[window['test_start']:window['test_end']]
            
            try:
                window_result = method_func(train_returns, test_returns)
                window_result['window_id'] = window['window_id']
                window_result['train_start'] = window['train_start']
                window_result['train_end'] = window['train_end']
                window_result['test_start'] = window['test_start']
                window_result['test_end'] = window['test_end']
                window_result['method'] = method_name
                results.append(window_result)
            except Exception as e:
                print(f"Error in window {window['window_id']}: {e}")
                continue
        
        return pd.DataFrame(results)

