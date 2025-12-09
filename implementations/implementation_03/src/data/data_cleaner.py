"""
Data cleaning pipeline for stock data.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class DataCleaner:
    """Cleans and aligns stock data."""
    
    def __init__(self, interpolation_method: str = 'time', drop_missing: bool = True):
        """
        Initialize the data cleaner.
        
        Args:
            interpolation_method: Method for interpolating missing values.
                                 Options: 'time', 'linear', 'forward_fill', 'backward_fill'
            drop_missing: If True, drop rows with any remaining missing values after interpolation
        """
        self.interpolation_method = interpolation_method
        self.drop_missing = drop_missing
    
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
            # Ensure index is DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, errors='coerce')
            
            # Remove rows with invalid dates
            df = df[df.index.notna()]
            
            # Sort by date
            df = df.sort_index()
            
            cleaned[symbol] = df
        
        return cleaned
    
    def align_trading_days(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Align all assets on common trading dates using inner join (intersection of dates).
        
        Args:
            data_dict: Dictionary mapping symbol to DataFrame
            
        Returns:
            Dictionary with aligned DataFrames (all have same date index)
        """
        if not data_dict:
            raise ValueError("No data to align")
        
        # Get intersection of all dates (common trading days)
        all_dates = None
        for symbol, df in data_dict.items():
            if all_dates is None:
                all_dates = set(df.index)
            else:
                all_dates = all_dates.intersection(set(df.index))
        
        if not all_dates:
            raise ValueError("No common trading days found across all stocks")
        
        all_dates = sorted(list(all_dates))
        
        # Create aligned DataFrames
        aligned_dict = {}
        for symbol, df in data_dict.items():
            aligned_df = df.loc[all_dates].copy()
            aligned_dict[symbol] = aligned_df
        
        print(f"Aligned data on {len(all_dates)} common trading days")
        print(f"Date range: {all_dates[0]} to {all_dates[-1]}")
        
        return aligned_dict
    
    def handle_missing_values(
        self, 
        data_dict: Dict[str, pd.DataFrame],
        method: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Handle missing values in the data.
        
        Args:
            data_dict: Dictionary mapping symbol to DataFrame
            method: Interpolation method (overrides self.interpolation_method if provided)
            
        Returns:
            Dictionary with DataFrames with missing values handled
        """
        method = method or self.interpolation_method
        cleaned = {}
        
        for symbol, df in data_dict.items():
            df_clean = df.copy()
            
            # Count missing values before cleaning
            missing_before = df_clean.isna().sum().sum()
            
            if missing_before > 0:
                if method == 'time':
                    # Time-based interpolation
                    df_clean = df_clean.interpolate(method='time', limit_direction='both')
                elif method == 'linear':
                    # Linear interpolation
                    df_clean = df_clean.interpolate(method='linear', limit_direction='both')
                elif method == 'forward_fill':
                    # Forward fill
                    df_clean = df_clean.fillna(method='ffill')
                elif method == 'backward_fill':
                    # Backward fill
                    df_clean = df_clean.fillna(method='bfill')
                else:
                    raise ValueError(f"Unknown interpolation method: {method}")
                
                # Count missing values after cleaning
                missing_after = df_clean.isna().sum().sum()
                if missing_after > 0:
                    print(f"  {symbol}: {missing_before} missing values before, {missing_after} after {method} interpolation")
            
            cleaned[symbol] = df_clean
        
        return cleaned
    
    def drop_residual_missing(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Drop any remaining rows with missing values after interpolation.
        
        Args:
            data_dict: Dictionary mapping symbol to DataFrame
            
        Returns:
            Dictionary with cleaned DataFrames
        """
        if not self.drop_missing:
            return data_dict
        
        # Find common dates with no missing values across all stocks
        all_valid_dates = None
        
        for symbol, df in data_dict.items():
            # Get dates with no missing values for this stock
            valid_dates = set(df.dropna().index)
            
            if all_valid_dates is None:
                all_valid_dates = valid_dates
            else:
                all_valid_dates = all_valid_dates.intersection(valid_dates)
        
        if not all_valid_dates:
            print("Warning: No dates with complete data across all stocks")
            return data_dict
        
        all_valid_dates = sorted(list(all_valid_dates))
        
        # Filter all DataFrames to common valid dates
        cleaned = {}
        for symbol, df in data_dict.items():
            cleaned[symbol] = df.loc[all_valid_dates].copy()
        
        print(f"Dropped rows with missing values. Remaining: {len(all_valid_dates)} trading days")
        
        return cleaned
    
    def validate_data(self, data_dict: Dict[str, pd.DataFrame]) -> bool:
        """
        Validate that data is clean and consistent.
        
        Args:
            data_dict: Dictionary mapping symbol to DataFrame
            
        Returns:
            True if validation passes, raises ValueError otherwise
        """
        if not data_dict:
            raise ValueError("No data to validate")
        
        # Check all DataFrames have same index
        first_symbol = list(data_dict.keys())[0]
        first_index = data_dict[first_symbol].index
        
        for symbol, df in data_dict.items():
            if not df.index.equals(first_index):
                raise ValueError(f"{symbol} has different date index than {first_symbol}")
            
            # Check for missing values
            if df.isna().any().any():
                missing_cols = df.columns[df.isna().any()].tolist()
                raise ValueError(f"{symbol} has missing values in columns: {missing_cols}")
            
            # Check for negative prices/volumes
            price_cols = ['Open', 'High', 'Low', 'Close', 'Price']
            for col in price_cols:
                if col in df.columns:
                    if (df[col] < 0).any():
                        raise ValueError(f"{symbol} has negative values in {col}")
            
            if 'Volume' in df.columns:
                if (df['Volume'] < 0).any():
                    raise ValueError(f"{symbol} has negative values in Volume")
            
            # Check OHLC consistency
            if all(col in df.columns for col in ['High', 'Low', 'Open', 'Close']):
                invalid = (df['High'] < df['Low']).any() or \
                         (df['High'] < df['Open']).any() or \
                         (df['High'] < df['Close']).any() or \
                         (df['Low'] > df['Open']).any() or \
                         (df['Low'] > df['Close']).any()
                
                if invalid:
                    print(f"Warning: {symbol} has some OHLC inconsistencies (may be due to data adjustments)")
        
        print("Data validation passed")
        return True
    
    def clean(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Run the complete cleaning pipeline.
        
        Args:
            data_dict: Dictionary mapping symbol to DataFrame
            
        Returns:
            Dictionary mapping symbol to cleaned DataFrame
        """
        print("Starting data cleaning pipeline...")
        
        # Step 1: Parse dates and sort
        print("\nStep 1: Parsing dates and sorting...")
        data_dict = self.parse_dates_and_sort(data_dict)
        
        # Step 2: Align trading days
        print("\nStep 2: Aligning trading days...")
        data_dict = self.align_trading_days(data_dict)
        
        # Step 3: Handle missing values
        print(f"\nStep 3: Handling missing values using {self.interpolation_method} method...")
        data_dict = self.handle_missing_values(data_dict)
        
        # Step 4: Drop residual missing (if enabled)
        if self.drop_missing:
            print("\nStep 4: Dropping rows with residual missing values...")
            data_dict = self.drop_residual_missing(data_dict)
        
        # Step 5: Validate data
        print("\nStep 5: Validating cleaned data...")
        self.validate_data(data_dict)
        
        print("\nData cleaning pipeline completed successfully!")
        
        return data_dict

