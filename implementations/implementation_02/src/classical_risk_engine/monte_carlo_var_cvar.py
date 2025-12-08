"""Monte Carlo VaR and CVaR calculation."""
import numpy as np
import pandas as pd
from typing import Optional
from scipy import stats
import time


class MonteCarloVaRCVaR:
    """Monte Carlo Simulation for VaR and CVaR."""
    
    def __init__(self, num_mc_paths: int = 100000, random_seed: Optional[int] = None):
        """
        Initialize Monte Carlo VaR/CVaR calculator.
        
        Args:
            num_mc_paths: Number of Monte Carlo simulation paths
            random_seed: Random seed for reproducibility
        """
        self.num_mc_paths = num_mc_paths
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def calculate(
        self,
        returns_df: pd.DataFrame,
        portfolio_weights: np.ndarray,
        confidence_levels: list = [0.95, 0.99],
        risk_horizon_days: int = 1
    ) -> dict:
        """
        Calculate VaR and CVaR using Monte Carlo simulation.
        
        Args:
            returns_df: DataFrame with returns (Date x Symbol)
            portfolio_weights: Portfolio weights array
            confidence_levels: List of confidence levels
            risk_horizon_days: Risk horizon in days
            
        Returns:
            Dictionary with VaR/CVaR estimates, runtime stats, and diagnostics
        """
        start_time = time.time()
        
        # Calculate mean and covariance
        mean_returns = returns_df.mean().values
        cov_matrix = returns_df.cov().values
        
        # Annualize
        mean_returns = mean_returns * 252
        cov_matrix = cov_matrix * 252
        
        # Scale for horizon
        horizon_factor = risk_horizon_days / 252
        mean_returns = mean_returns * horizon_factor
        cov_matrix = cov_matrix * horizon_factor
        
        # Simulate returns
        simulated_returns = np.random.multivariate_normal(
            mean_returns,
            cov_matrix,
            size=self.num_mc_paths
        )
        
        # Calculate portfolio returns
        portfolio_sim_returns = np.dot(simulated_returns, portfolio_weights)
        
        # Calculate VaR and CVaR for each confidence level
        var_estimates = {}
        cvar_estimates = {}
        
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            var = -np.percentile(portfolio_sim_returns, alpha * 100)
            var_estimates[conf_level] = var
            
            # CVaR: expected value of losses beyond VaR
            tail_losses = portfolio_sim_returns[portfolio_sim_returns <= -var]
            if len(tail_losses) > 0:
                cvar = -np.mean(tail_losses)
            else:
                cvar = var
            cvar_estimates[conf_level] = cvar
        
        runtime = time.time() - start_time
        
        return {
            'var_mc_estimates': var_estimates,
            'cvar_mc_estimates': cvar_estimates,
            'runtime_stats': {
                'wall_clock_time': runtime,
                'num_simulations': self.num_mc_paths,
                'method': 'monte_carlo_var_cvar'
            },
            'diagnostics': {
                'simulated_returns_mean': np.mean(portfolio_sim_returns),
                'simulated_returns_std': np.std(portfolio_sim_returns),
                'min_simulated_return': np.min(portfolio_sim_returns),
                'max_simulated_return': np.max(portfolio_sim_returns)
            }
        }

