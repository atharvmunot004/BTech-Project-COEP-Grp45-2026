"""Data cleaning pipeline."""
import pandas as pd
import numpy as np
from typing import Dict, List
from config.config_loader import get_config


class DataCleaner:
    """Cleans and aligns stock data."""
    
    def __init__(self):
        """Initialize the data cleaner."""
        self.config = get_config()
        self.cleaning_config = self.config.get_data_config().get('cleaning_pipeline', {})
    
    def parse_dates_and_sort(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Parse date column, set as index, and sort ascending by date.
        
        Args:
            data_dict: Dictionary mapping symbol to DataFrame
            
        Returns:
            Dictionary with sorted DataFrames
        """
        cleaned = {}
        for symbol, df in data_dict.items():
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, errors='coerce')
            df = df.sort_index()
            cleaned[symbol] = df
        return cleaned
    
    def align_trading_days(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Align all assets on common trading dates using inner join on dates.
        
        Args:
            data_dict: Dictionary mapping symbol to DataFrame
            
        Returns:
            Panel DataFrame with MultiIndex (Date, Symbol) or wide format
        """
        if not data_dict:
            raise ValueError("No data to align")
        
        # Get all dates (intersection)
        all_dates = None
        for symbol, df in data_dict.items():
            if all_dates is None:
                all_dates = set(df.index)
            else:
                all_dates = all_dates.intersection(set(df.index))
        
        all_dates = sorted(list(all_dates))
        
        # Create aligned DataFrames
        aligned_dict = {}
        for symbol, df in data_dict.items():
            aligned_df = df.loc[all_dates].copy()
            aligned_dict[symbol] = aligned_df
        
        # Create wide format panel: Date x Symbol for each column
        # We'll use adj_close for returns calculation
        panel_data = {}
        for col in ['Open', 'High', 'Low', 'Close', 'adj_close', 'Volume']:
            panel_data[col] = pd.DataFrame({
                symbol: df[col] for symbol, df in aligned_dict.items()
            })
        
        return panel_data
    
    def handle_missing_values_interpolation(
        self, 
        panel_data: Dict[str, pd.DataFrame],
        strategy: str = "time_interpolation"
    ) -> Dict[str, pd.DataFrame]:
        """
        Interpolate missing OHLCV values per asset along the time axis.
        
        Args:
            panel_data: Dictionary mapping column name to DataFrame (Date x Symbol)
            strategy: Interpolation strategy ('time_interpolation' or 'linear')
            
        Returns:
            Dictionary with interpolated DataFrames
        """
        interpolated = {}
        for col, df in panel_data.items():
            if strategy == "time_interpolation":
                # Time-based interpolation
                df_interp = df.interpolate(method='time', limit_direction='both')
            else:
                # Linear interpolation
                df_interp = df.interpolate(method='linear', limit_direction='both')
            interpolated[col] = df_interp
        return interpolated
    
    def drop_residual_missing(self, panel_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Drop any remaining rows with missing values after interpolation.
        
        Args:
            panel_data: Dictionary mapping column name to DataFrame
            
        Returns:
            Dictionary with cleaned DataFrames
        """
        cleaned = {}
        for col, df in panel_data.items():
            df_clean = df.dropna()
            cleaned[col] = df_clean
        return cleaned
    
    def clean(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Run the complete cleaning pipeline.
        
        Args:
            data_dict: Dictionary mapping symbol to DataFrame
            
        Returns:
            Dictionary mapping column name to aligned DataFrame (Date x Symbol)
        """
        # Step 1: Parse dates and sort
        data_dict = self.parse_dates_and_sort(data_dict)
        
        # Step 2: Align trading days
        panel_data = self.align_trading_days(data_dict)
        
        # Step 3: Handle missing values
        strategy = self.cleaning_config.get('steps', [{}])[2].get('strategy', 'time_interpolation')
        panel_data = self.handle_missing_values_interpolation(panel_data, strategy)
        
        # Step 4: Drop residual missing
        panel_data = self.drop_residual_missing(panel_data)
        
        return panel_data

