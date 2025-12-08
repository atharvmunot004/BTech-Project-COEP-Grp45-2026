"""Returns calculation module."""
import pandas as pd
import numpy as np
from typing import Dict
from config.config_loader import get_config


class ReturnsCalculator:
    """Calculates returns from price data."""
    
    def __init__(self):
        """Initialize the returns calculator."""
        self.config = get_config()
        returns_config = self.config.get_data_config().get('returns_config', {})
        self.price_field = returns_config.get('price_field', 'adj_close')
        self.return_type = returns_config.get('return_type', 'log')
        self.risk_horizon_days = returns_config.get('risk_horizon_days', 1)
    
    def calculate_returns(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate returns from prices.
        
        Args:
            prices_df: DataFrame with Date index and Symbol columns (prices)
            
        Returns:
            DataFrame with returns (Date x Symbol)
        """
        if self.return_type == 'log':
            # Log returns: r_t = ln(P_t / P_{t-1})
            returns = np.log(prices_df / prices_df.shift(self.risk_horizon_days))
        else:
            # Simple returns: r_t = (P_t / P_{t-1}) - 1
            returns = (prices_df / prices_df.shift(self.risk_horizon_days)) - 1
        
        # Drop first row (NaN)
        returns = returns.dropna()
        
        return returns
    
    def calculate_mean_returns(self, returns_df: pd.DataFrame, annualize: bool = True) -> pd.Series:
        """
        Calculate mean returns per asset.
        
        Args:
            returns_df: DataFrame with returns (Date x Symbol)
            annualize: Whether to annualize returns (assumes daily data)
            
        Returns:
            Series with mean returns per symbol
        """
        mean_returns = returns_df.mean()
        if annualize:
            # Annualize: multiply by 252 trading days
            mean_returns = mean_returns * 252
        return mean_returns
    
    def calculate_covariance_matrix(self, returns_df: pd.DataFrame, annualize: bool = True) -> pd.DataFrame:
        """
        Calculate covariance matrix.
        
        Args:
            returns_df: DataFrame with returns (Date x Symbol)
            annualize: Whether to annualize covariance (assumes daily data)
            
        Returns:
            Covariance matrix (Symbol x Symbol)
        """
        cov_matrix = returns_df.cov()
        if annualize:
            # Annualize: multiply by 252 trading days
            cov_matrix = cov_matrix * 252
        return cov_matrix
    
    def calculate_correlation_matrix(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation matrix.
        
        Args:
            returns_df: DataFrame with returns (Date x Symbol)
            
        Returns:
            Correlation matrix (Symbol x Symbol)
        """
        return returns_df.corr()
    
    def calculate_volatilities(self, returns_df: pd.DataFrame, annualize: bool = True) -> pd.Series:
        """
        Calculate volatilities (standard deviations) per asset.
        
        Args:
            returns_df: DataFrame with returns (Date x Symbol)
            annualize: Whether to annualize volatility (assumes daily data)
            
        Returns:
            Series with volatilities per symbol
        """
        volatilities = returns_df.std()
        if annualize:
            # Annualize: multiply by sqrt(252) for daily data
            volatilities = volatilities * np.sqrt(252)
        return volatilities

