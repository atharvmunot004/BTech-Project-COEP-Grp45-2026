"""
Complete preprocessing pipeline: load data, clean, align, calculate features.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from data.load_data import DataLoader
from data.calculate_returns import calculate_returns, calculate_volatilities
from data.calculate_covariances import calculate_covariance_matrix, calculate_correlation_matrix, calculate_mean_returns


class PreprocessingPipeline:
    """Complete preprocessing pipeline for portfolio analysis."""
    
    def __init__(self, dataset_path: Optional[str] = None):
        """Initialize the preprocessing pipeline."""
        # Handle path resolution
        if dataset_path is None:
            # Try to find dataset folder relative to project
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent.parent.parent
            dataset_path = str(project_root / "dataset")
        self.loader = DataLoader(dataset_path)
        self.prices_df: Optional[pd.DataFrame] = None
        self.returns_df: Optional[pd.DataFrame] = None
        self.covariance_matrix: Optional[pd.DataFrame] = None
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.mean_returns: Optional[pd.Series] = None
        self.volatilities: Optional[pd.Series] = None
    
    def load_and_preprocess(
        self,
        symbols: List[str],
        return_method: str = "log",
        min_date: Optional[str] = None,
        max_date: Optional[str] = None
    ) -> Dict:
        """
        Load and preprocess data for given symbols.
        
        Args:
            symbols: List of stock symbols to load
            return_method: "simple" or "log" returns
            min_date: Minimum date (YYYY-MM-DD) to include
            max_date: Maximum date (YYYY-MM-DD) to include
            
        Returns:
            Dictionary with all preprocessed data
        """
        # Load prices
        print(f"Loading data for {len(symbols)} symbols...")
        self.prices_df = self.loader.load_multiple_stocks(symbols)
        
        # Filter by date range if specified
        if min_date:
            self.prices_df = self.prices_df[self.prices_df.index >= min_date]
        if max_date:
            self.prices_df = self.prices_df[self.prices_df.index <= max_date]
        
        # Align data (forward fill missing values)
        self.prices_df = self.prices_df.ffill().bfill()
        
        # Calculate returns
        print("Calculating returns...")
        self.returns_df = calculate_returns(self.prices_df, method=return_method)
        
        # Calculate statistics
        print("Calculating statistics...")
        self.mean_returns = calculate_mean_returns(self.returns_df)
        self.covariance_matrix = calculate_covariance_matrix(self.returns_df)
        self.correlation_matrix = calculate_correlation_matrix(self.returns_df)
        self.volatilities = calculate_volatilities(self.returns_df)
        
        return {
            "prices_df": self.prices_df,
            "returns_df": self.returns_df,
            "mean_returns": self.mean_returns,
            "covariance_matrix": self.covariance_matrix,
            "correlation_matrix": self.correlation_matrix,
            "volatilities": self.volatilities,
            "symbols": symbols
        }
    
    def get_summary_stats(self) -> pd.DataFrame:
        """Get summary statistics for the preprocessed data."""
        if self.returns_df is None:
            raise ValueError("Data not preprocessed yet. Call load_and_preprocess first.")
        
        summary = pd.DataFrame({
            "Mean Return (annualized)": self.mean_returns,
            "Volatility (annualized)": self.volatilities,
            "Sharpe Ratio": self.mean_returns / self.volatilities,
            "Min Return": self.returns_df.min(),
            "Max Return": self.returns_df.max(),
            "Skewness": self.returns_df.skew(),
            "Kurtosis": self.returns_df.kurtosis()
        })
        
        return summary


if __name__ == "__main__":
    # Test the pipeline
    pipeline = PreprocessingPipeline()
    
    # Get all available symbols
    all_symbols = pipeline.loader.get_available_symbols()
    print(f"Available symbols: {all_symbols}")
    
    # Preprocess a subset
    symbols = all_symbols[:5]  # First 5 symbols
    data = pipeline.load_and_preprocess(symbols)
    
    print("\nPrices shape:", data["prices_df"].shape)
    print("\nReturns shape:", data["returns_df"].shape)
    print("\nMean returns:")
    print(data["mean_returns"])
    print("\nCovariance matrix shape:", data["covariance_matrix"].shape)
    
    print("\nSummary statistics:")
    print(pipeline.get_summary_stats())

