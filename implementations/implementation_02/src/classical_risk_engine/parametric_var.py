"""Parametric VaR calculation."""
import numpy as np
import pandas as pd
from typing import Union, Optional
from scipy import stats
import time


class ParametricVaR:
    """Variance-Covariance (Parametric) Value-at-Risk."""
    
    def __init__(self, confidence_levels: list = [0.95, 0.99], use_historical_mean: bool = True):
        """
        Initialize Parametric VaR calculator.
        
        Args:
            confidence_levels: List of confidence levels (e.g., [0.95, 0.99])
            use_historical_mean: Whether to use historical mean (True) or assume zero mean
        """
        self.confidence_levels = confidence_levels
        self.use_historical_mean = use_historical_mean
    
    def calculate(
        self,
        returns_df: pd.DataFrame,
        portfolio_weights: np.ndarray,
        risk_horizon_days: int = 1
    ) -> dict:
        """
        Calculate Parametric VaR for a portfolio.
        
        Args:
            returns_df: DataFrame with returns (Date x Symbol)
            portfolio_weights: Portfolio weights array
            risk_horizon_days: Risk horizon in days
            
        Returns:
            Dictionary with VaR estimates, runtime stats, and diagnostics
        """
        start_time = time.time()
        
        # Calculate mean and covariance
        mean_returns = returns_df.mean().values
        cov_matrix = returns_df.cov().values
        
        # Annualize
        mean_returns = mean_returns * 252
        cov_matrix = cov_matrix * 252
        
        # Portfolio statistics
        portfolio_mean = np.dot(portfolio_weights, mean_returns)
        portfolio_variance = np.dot(portfolio_weights, np.dot(cov_matrix, portfolio_weights))
        portfolio_std = np.sqrt(portfolio_variance)
        
        # Calculate VaR for each confidence level
        var_estimates = {}
        for conf_level in self.confidence_levels:
            alpha = 1 - conf_level
            z_score = stats.norm.ppf(conf_level)
            
            if self.use_historical_mean:
                var = -portfolio_mean * (risk_horizon_days / 252) - z_score * portfolio_std * np.sqrt(risk_horizon_days / 252)
            else:
                var = -z_score * portfolio_std * np.sqrt(risk_horizon_days / 252)
            
            var_estimates[conf_level] = var
        
        runtime = time.time() - start_time
        
        return {
            'var_estimates': var_estimates,
            'runtime_stats': {
                'wall_clock_time': runtime,
                'method': 'parametric_var'
            },
            'diagnostics': {
                'portfolio_mean': portfolio_mean,
                'portfolio_std': portfolio_std,
                'portfolio_variance': portfolio_variance
            }
        }

