"""
Data Layer: Load and serve raw OHLCV data from CSV files.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import os


class DataLoader:
    """Loads stock data from CSV files in the dataset folder."""
    
    def __init__(self, dataset_path: str = None):
        """
        Initialize the data loader.
        
        Args:
            dataset_path: Path to the dataset folder. If None, uses relative path.
        """
        if dataset_path is None:
            # Default to dataset folder relative to project root
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent.parent.parent
            self.dataset_path = project_root / "dataset"
        else:
            self.dataset_path = Path(dataset_path)
        
        if not self.dataset_path.exists():
            # Try alternative path structure
            alt_path = Path(__file__).parent.parent.parent.parent.parent / "dataset"
            if alt_path.exists():
                self.dataset_path = alt_path
            else:
                raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")
    
    def load_single_stock(self, symbol: str) -> pd.DataFrame:
        """
        Load a single stock's data from CSV.
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE', 'TCS')
            
        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume
            Index is Date
        """
        csv_file = self.dataset_path / f"{symbol}_10yr_daily.csv"
        
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        
        # Read CSV - the format has headers on multiple rows
        # Row 1: Price,Close,High,Low,Open,Volume (column names)
        # Row 2: Ticker,RELIANCE.NS,... (ticker info)
        # Row 3: Date,,,,, (date label)
        # Row 4+: Actual data (Date,Price,Close,High,Low,Open,Volume)
        
        # Read the CSV properly
        # First row has column names: Price,Close,High,Low,Open,Volume
        # Third row starts the data with Date as first column
        df = pd.read_csv(csv_file, skiprows=2)  # Skip header and ticker rows
        
        # The first column should be Date, rest are Price,Close,High,Low,Open,Volume
        # But after skiprows=2, pandas uses the third row as header which is "Date,,,,,"
        # So we need to manually set column names
        
        # Read first line to get actual column names
        with open(csv_file, 'r') as f:
            first_line = f.readline().strip()
            actual_columns = first_line.split(',')
        
        # Read data starting from row 4 (index 3), with proper column names
        df = pd.read_csv(csv_file, skiprows=3, header=None)
        
        # Set column names: first is Date, then Price,Close,High,Low,Open,Volume
        if len(actual_columns) == len(df.columns):
            df.columns = actual_columns
        else:
            # Fallback: use standard names
            df.columns = ['Date'] + actual_columns[1:] if len(df.columns) == len(actual_columns) else ['Date', 'Price', 'Close', 'High', 'Low', 'Open', 'Volume'][:len(df.columns)]
        
        # Ensure Date column exists and is datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])  # Remove rows where date parsing failed
            df.set_index('Date', inplace=True)
        else:
            # First column should be date
            first_col = df.columns[0]
            df[first_col] = pd.to_datetime(df[first_col], errors='coerce')
            df = df.dropna(subset=[first_col])
            df.set_index(first_col, inplace=True)
        
        # Fix index name if it's not 'Date'
        if df.index.name != 'Date':
            df.index.name = 'Date'
        
        # Handle Price vs Close - prefer Close, use Price if Close missing
        if 'Price' in df.columns and 'Close' not in df.columns:
            df['Close'] = df['Price']
        elif 'Price' in df.columns and 'Close' in df.columns:
            # If both exist and Close has NaN, fill from Price
            if df['Close'].isna().any():
                df['Close'] = df['Close'].fillna(df['Price'])
            # Drop Price column
            df = df.drop(columns=['Price'])
        
        # Select and order columns we need
        columns_needed = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_cols = [col for col in columns_needed if col in df.columns]
        if 'Close' not in available_cols:
            # Close is essential, try to find it
            if 'Price' in df.columns:
                df['Close'] = df['Price']
                available_cols.append('Close')
        df = df[available_cols]
        
        # Sort by date
        df.sort_index(inplace=True)
        
        return df
    
    def load_multiple_stocks(self, symbols: List[str]) -> pd.DataFrame:
        """
        Load multiple stocks and return a panel DataFrame.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            DataFrame with MultiIndex (Date, Symbol) or wide format (Date x Symbol)
        """
        data_dict = {}
        
        for symbol in symbols:
            try:
                df = self.load_single_stock(symbol)
                data_dict[symbol] = df
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                continue
        
        if not data_dict:
            raise ValueError("No valid stock data loaded")
        
        # Create wide format: Date x Symbol for Close prices
        close_prices = pd.DataFrame({
            symbol: df['Close'] for symbol, df in data_dict.items()
        })
        
        return close_prices
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available stock symbols from CSV files."""
        csv_files = list(self.dataset_path.glob("*_10yr_daily.csv"))
        symbols = [f.stem.replace("_10yr_daily", "") for f in csv_files]
        return sorted(symbols)


if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()
    print("Available symbols:", loader.get_available_symbols())
    
    # Load a single stock
    reliance = loader.load_single_stock("RELIANCE")
    print("\nRELIANCE data shape:", reliance.shape)
    print(reliance.head())
    
    # Load multiple stocks
    symbols = ["RELIANCE", "TCS", "HDFCBANK"]
    prices = loader.load_multiple_stocks(symbols)
    print("\nMulti-stock prices shape:", prices.shape)
    print(prices.head())

