"""Extreme Value Theory - Peaks Over Threshold (POT)."""
import numpy as np
import pandas as pd
from typing import Optional
from scipy import stats
import time
try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class EVTPOT:
    """Extreme Value Theory using Peaks Over Threshold method."""
    
    def __init__(self, threshold_type: str = 'quantile', threshold_quantile: float = 0.95):
        """
        Initialize EVT POT calculator.
        
        Args:
            threshold_type: Type of threshold ('quantile' or 'fixed')
            threshold_quantile: Quantile for threshold selection (e.g., 0.95)
        """
        self.threshold_type = threshold_type
        self.threshold_quantile = threshold_quantile
    
    def _fit_gpd(self, excesses: np.ndarray) -> tuple:
        """Fit Generalized Pareto Distribution to excesses."""
        if len(excesses) < 10:
            # Not enough data, use method of moments
            mean_excess = np.mean(excesses)
            var_excess = np.var(excesses)
            if var_excess > 0:
                shape = 0.5 * (1 - (mean_excess ** 2) / var_excess)
                scale = mean_excess * (1 - shape)
            else:
                shape = 0.0
                scale = mean_excess
            return shape, scale
        
        # Maximum likelihood estimation
        def neg_log_likelihood(params):
            shape, scale = params
            if scale <= 0:
                return 1e10
            if shape < -0.5:
                return 1e10
            try:
                if shape == 0:
                    # Exponential distribution
                    ll = -len(excesses) * np.log(scale) - np.sum(excesses) / scale
                else:
                    # GPD
                    z = 1 + shape * excesses / scale
                    if np.any(z <= 0):
                        return 1e10
                    ll = -len(excesses) * np.log(scale) - (1 + 1/shape) * np.sum(np.log(z))
                return -ll
            except:
                return 1e10
        
        # Initial guess
        mean_excess = np.mean(excesses)
        initial_shape = 0.1
        initial_scale = mean_excess * (1 - initial_shape)
        
        try:
            result = minimize(neg_log_likelihood, [initial_shape, initial_scale], method='L-BFGS-B',
                            bounds=[(-0.5, 1.0), (1e-6, None)])
            shape, scale = result.x
        except:
            shape, scale = initial_shape, initial_scale
        
        return shape, scale
    
    def calculate(
        self,
        portfolio_loss_series: pd.Series,
        confidence_levels: list = [0.95, 0.99]
    ) -> dict:
        """
        Calculate VaR and CVaR using EVT POT method.
        
        Args:
            portfolio_loss_series: Series of portfolio losses (negative returns)
            confidence_levels: List of confidence levels
            
        Returns:
            Dictionary with VaR/CVaR estimates, runtime stats, and diagnostics
        """
        start_time = time.time()
        
        # Convert to losses (positive values)
        losses = -portfolio_loss_series.values  # Losses are positive
        
        # Select threshold
        if self.threshold_type == 'quantile':
            threshold = np.percentile(losses, self.threshold_quantile * 100)
        else:
            threshold = self.threshold_quantile
        
        # Get excesses over threshold
        excesses = losses[losses > threshold] - threshold
        
        if len(excesses) < 5:
            # Not enough excesses, use empirical method
            var_estimates = {}
            cvar_estimates = {}
            for conf_level in confidence_levels:
                alpha = 1 - conf_level
                var_estimates[conf_level] = np.percentile(losses, (1 - alpha) * 100)
                tail_losses = losses[losses >= var_estimates[conf_level]]
                cvar_estimates[conf_level] = np.mean(tail_losses) if len(tail_losses) > 0 else var_estimates[conf_level]
        else:
            # Fit GPD
            shape, scale = self._fit_gpd(excesses)
            
            # Calculate VaR and CVaR
            n = len(losses)
            n_u = len(excesses)
            var_estimates = {}
            cvar_estimates = {}
            
            for conf_level in confidence_levels:
                alpha = 1 - conf_level
                # VaR using GPD
                if shape != 0:
                    var = threshold + (scale / shape) * (((n / n_u) * alpha) ** (-shape) - 1)
                else:
                    var = threshold - scale * np.log((n / n_u) * alpha)
                
                var_estimates[conf_level] = var
                
                # CVaR using GPD
                if shape != 0 and shape < 1:
                    cvar = var / (1 - shape) + scale / (1 - shape)
                else:
                    cvar = var + scale
                
                cvar_estimates[conf_level] = cvar
        
        runtime = time.time() - start_time
        
        return {
            'var_evt_estimates': var_estimates,
            'cvar_evt_estimates': cvar_estimates,
            'runtime_stats': {
                'wall_clock_time': runtime,
                'method': 'evt_pot'
            },
            'diagnostics': {
                'threshold': threshold,
                'num_excesses': len(excesses),
                'gpd_shape': shape if len(excesses) >= 5 else None,
                'gpd_scale': scale if len(excesses) >= 5 else None
            }
        }

