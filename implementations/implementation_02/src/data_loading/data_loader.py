"""Data loader for CSV files."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from config.config_loader import get_config


class DataLoader:
    """Loads stock data from CSV files in the dataset folder."""
    
    def __init__(self, dataset_path: Optional[str] = None):
        """
        Initialize the data loader.
        
        Args:
            dataset_path: Path to the dataset folder. If None, uses config to find it.
        """
        if dataset_path is None:
            config = get_config()
            self.dataset_path = config.get_dataset_path()
        else:
            self.dataset_path = Path(dataset_path)
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")
    
    def load_single_stock(self, symbol: str) -> pd.DataFrame:
        """
        Load a single stock's data from CSV.
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE', 'TCS')
            
        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume (or adj_close if available)
            Index is Date
        """
        csv_file = self.dataset_path / f"{symbol}_10yr_daily.csv"
        
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        
        # Read CSV - skip first 3 rows (header, ticker, date label)
        # After skiprows=3, we have: Date, Price, Close, High, Low, Open (6 columns total)
        df = pd.read_csv(csv_file, skiprows=3, header=None)
        
        # Set column names based on actual structure
        # The CSV structure is: Date, Price, Close, High, Low, Open
        # Volume is missing in the actual data, so we have 6 columns
        if len(df.columns) == 6:
            df.columns = ['Date', 'Price', 'Close', 'High', 'Low', 'Open']
        else:
            # Fallback
            df.columns = ['Date', 'Price', 'Close', 'High', 'Low', 'Open', 'Volume'][:len(df.columns)]
        
        # Parse Date column
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df.set_index('Date', inplace=True)
        
        # Use Close as adj_close (data is already adjusted)
        df['adj_close'] = df['Close']
        
        # Add Volume column if missing (set to 0 or NaN)
        if 'Volume' not in df.columns:
            df['Volume'] = 0  # Placeholder
        
        # Select and order columns
        columns_needed = ['Open', 'High', 'Low', 'Close', 'adj_close', 'Volume']
        available_cols = [col for col in columns_needed if col in df.columns]
        df = df[available_cols]
        
        # Sort by date
        df.sort_index(inplace=True)
        
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
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                continue
        
        if not data_dict:
            raise ValueError("No valid stock data loaded")
        
        return data_dict
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available stock symbols from CSV files."""
        csv_files = list(self.dataset_path.glob("*_10yr_daily.csv"))
        symbols = [f.stem.replace("_10yr_daily", "") for f in csv_files]
        return sorted(symbols)

