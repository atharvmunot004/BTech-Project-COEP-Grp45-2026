"""
Data loader for CSV files from raw data directory.

This module loads stock data from CSV files following the structure defined in:
implementation_03/data/metadata/llm.json

File structure per metadata:
- header_row: 1 (Price, Close, High, Low, Open, Volume)
- ticker_row: 2 (Ticker info)
- date_row: 3 (Date label)
- data_start_row: 4 (Actual data: Date, Price, Close, High, Low, Open, Volume)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import json


class DataLoader:
    """Loads stock data from CSV files in the raw data folder."""
    
    def __init__(self, raw_data_path: Optional[Path] = None):
        """
        Initialize the data loader.
        
        Args:
            raw_data_path: Path to the raw data folder. If None, uses default relative path.
        """
        if raw_data_path is None:
            # Default path: implementation_03/data/raw
            # __file__ is: implementation_03/src/data/data_loader.py
            # We need to go up 3 levels to reach implementation_03
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent  # implementation_03
            self.raw_data_path = project_root / "data" / "raw"
        else:
            self.raw_data_path = Path(raw_data_path)
        
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Raw data path not found: {self.raw_data_path}")
    
    def load_single_stock(self, symbol: str) -> pd.DataFrame:
        """
        Load a single stock's data from CSV.
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE', 'TCS')
            
        Returns:
            DataFrame with columns: Open, High, Low, Close, Price, Volume
            Index is Date (DatetimeIndex)
        """
        csv_file = self.raw_data_path / f"{symbol}_10yr_daily.csv"
        
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        
        # Read CSV based on metadata structure from llm.json
        # File structure per metadata:
        #   - header_row: 1 (Price, Close, High, Low, Open, Volume - 6 columns)
        #   - ticker_row: 2 (Ticker info)
        #   - date_row: 3 (Date label)
        #   - data_start_row: 4 (Date, Price, Close, High, Low, Open, Volume)
        #
        # The CSV has:
        #   Row 1: Price,Close,High,Low,Open,Volume (6 column headers)
        #   Row 2: Ticker,ASIANPAINT.NS,... (ticker info)
        #   Row 3: Date,,,,, (date label row)
        #   Row 4+: Date,Price,Close,High,Low,Open,Volume (7 values: Date + 6 data columns)
        #
        # When we skip 3 rows, pandas reads row 4 as first data row
        # Row 4 has 6 comma-separated values (Date + 5 data columns, or all 6 data columns)
        # Per metadata, we expect: Date, Price, Close, High, Low, Open, Volume
        
        # Read CSV skipping first 3 rows (header, ticker, date label)
        df = pd.read_csv(csv_file, skiprows=3, header=None)
        
        # Check how many columns pandas actually read
        num_cols = len(df.columns)
        
        # Per metadata, expected columns are: Price, Close, High, Low, Open, Volume (6 data columns)
        # Plus Date as index column = 7 total columns expected
        # But pandas may read only 6 columns if the CSV structure doesn't match exactly
        
        if num_cols == 6:
            # Pandas reads 6 columns. Based on metadata structure:
            # Column order should be: Date, Price, Close, High, Low, Open, Volume
            # But we only have 6 columns. Need to determine which column is missing.
            
            # Check if last column is Volume (large integers) or Open (price-like values)
            last_col_idx = num_cols - 1
            last_col_values = df.iloc[:, last_col_idx]
            
            # Volume typically has large integer values (> 1000)
            is_volume = (pd.api.types.is_integer_dtype(last_col_values) or 
                        pd.api.types.is_float_dtype(last_col_values)) and \
                       last_col_values.min() > 1000 and \
                       last_col_values.max() > 10000
            
            if is_volume:
                # Last column is Volume, so Open is missing
                # Assign columns: Date, Price, Close, High, Low, Volume
                df.columns = ['Date', 'Price', 'Close', 'High', 'Low', 'Volume']
                # Insert Open column - use average of High and Low as approximation
                df.insert(5, 'Open', (df['High'] + df['Low']) / 2)
            else:
                # Last column is likely Open, Volume is missing
                df.columns = ['Date', 'Price', 'Close', 'High', 'Low', 'Open']
                # Add Volume column with zeros (metadata indicates Volume should exist)
                df['Volume'] = 0
                
        elif num_cols == 7:
            # Perfect! We have all 7 columns as expected
            df.columns = ['Date', 'Price', 'Close', 'High', 'Low', 'Open', 'Volume']
        else:
            # Unexpected number of columns - try to map what we have
            # Per metadata, expected order: Date, Price, Close, High, Low, Open, Volume
            column_names = ['Date', 'Price', 'Close', 'High', 'Low', 'Open', 'Volume']
            df.columns = column_names[:num_cols]
            
            # Ensure all expected columns exist (add missing ones)
            expected_data_cols = ['Price', 'Close', 'High', 'Low', 'Open', 'Volume']
            for col in expected_data_cols:
                if col not in df.columns:
                    if col == 'Volume':
                        df['Volume'] = 0
                    elif col == 'Open':
                        # Use average of High and Low if available, else use Close
                        if 'High' in df.columns and 'Low' in df.columns:
                            df['Open'] = (df['High'] + df['Low']) / 2
                        elif 'Close' in df.columns:
                            df['Open'] = df['Close']
                        else:
                            df['Open'] = 0
        
        # Parse Date column
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df.set_index('Date', inplace=True)
        
        # Ensure we have all expected columns per metadata
        # Expected columns per llm.json: Price, Close, High, Low, Open, Volume
        expected_cols = ['Price', 'Close', 'High', 'Low', 'Open', 'Volume']
        
        # Verify all expected columns exist
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns for {symbol}: {missing_cols}")
        
        # Select columns in the order specified by metadata
        available_cols = [col for col in expected_cols if col in df.columns]
        df = df[available_cols]
        
        # Convert numeric columns
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by date
        df.sort_index(inplace=True)
        
        # Remove any duplicate dates (keep first occurrence)
        df = df[~df.index.duplicated(keep='first')]
        
        return df
    
    def load_multiple_stocks(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Load multiple stocks and return a dictionary.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        data_dict = {}
        
        for symbol in symbols:
            try:
                df = self.load_single_stock(symbol)
                data_dict[symbol] = df
                print(f"Loaded {symbol}: {len(df)} rows, date range: {df.index.min()} to {df.index.max()}")
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                continue
            except Exception as e:
                print(f"Error loading {symbol}: {e}")
                continue
        
        if not data_dict:
            raise ValueError("No valid stock data loaded")
        
        return data_dict
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available stock symbols from CSV files.
        
        Returns:
            List of stock symbols
        """
        csv_files = list(self.raw_data_path.glob("*_10yr_daily.csv"))
        symbols = [f.stem.replace("_10yr_daily", "") for f in csv_files]
        return sorted(symbols)
    
    def load_all_stocks(self) -> Dict[str, pd.DataFrame]:
        """
        Load all available stocks.
        
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        symbols = self.get_available_symbols()
        return self.load_multiple_stocks(symbols)

