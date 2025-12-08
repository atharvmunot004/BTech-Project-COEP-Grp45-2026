"""Metrics calculation for evaluation."""
import numpy as np
import pandas as pd
from typing import Dict, Optional, List


class MetricsCalculator:
    """Calculate performance and risk metrics."""
    
    @staticmethod
    def calculate_portfolio_metrics(
        returns: pd.Series,
        risk_free_rate: float = 0.0
    ) -> Dict:
        """
        Calculate portfolio performance metrics.
        
        Args:
            returns: Portfolio returns series
            risk_free_rate: Risk-free rate (annualized)
            
        Returns:
            Dictionary with metrics
        """
        # Annualize returns
        annualized_return = returns.mean() * 252
        annualized_volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else annualized_volatility
        sortino_ratio = (annualized_return - risk_free_rate) / downside_std if downside_std > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio
        }
    
    @staticmethod
    def calculate_risk_metrics(
        returns: pd.Series,
        var_estimates: Optional[Dict] = None,
        cvar_estimates: Optional[Dict] = None,
        confidence_levels: List[float] = [0.95, 0.99]
    ) -> Dict:
        """
        Calculate risk metrics including VaR/CVaR coverage.
        
        Args:
            returns: Portfolio returns series
            var_estimates: Dictionary of VaR estimates by confidence level
            cvar_estimates: Dictionary of CVaR estimates by confidence level
            confidence_levels: List of confidence levels
            
        Returns:
            Dictionary with risk metrics
        """
        metrics = {}
        
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            
            # Actual VaR/CVaR from realized returns
            actual_var = -np.percentile(returns, alpha * 100)
            tail_losses = returns[returns <= -actual_var]
            actual_cvar = -np.mean(tail_losses) if len(tail_losses) > 0 else actual_var
            
            metrics[f'actual_var_{conf_level}'] = actual_var
            metrics[f'actual_cvar_{conf_level}'] = actual_cvar
            
            # Coverage ratio
            if var_estimates and conf_level in var_estimates:
                predicted_var = var_estimates[conf_level]
                breaches = (returns <= -predicted_var).sum()
                total = len(returns)
                coverage_ratio = breaches / total if total > 0 else 0
                metrics[f'var_coverage_ratio_{conf_level}'] = coverage_ratio
                metrics[f'var_breaches_{conf_level}'] = breaches
            
            # CVaR accuracy
            if cvar_estimates and conf_level in cvar_estimates:
                predicted_cvar = cvar_estimates[conf_level]
                mae = abs(actual_cvar - predicted_cvar)
                rmse = np.sqrt((actual_cvar - predicted_cvar) ** 2)
                metrics[f'cvar_mae_{conf_level}'] = mae
                metrics[f'cvar_rmse_{conf_level}'] = rmse
        
        return metrics

