"""GARCH volatility forecasting."""
import numpy as np
import pandas as pd
from typing import Optional
import time
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    print("Warning: arch package not available. Install with: pip install arch")


class GARCHVolatility:
    """GARCH(1,1) Volatility Forecasting."""
    
    def __init__(self, distribution: str = 'normal'):
        """
        Initialize GARCH volatility forecaster.
        
        Args:
            distribution: Distribution assumption ('normal', 't', 'skewt')
        """
        if not ARCH_AVAILABLE:
            raise ImportError("arch package required. Install with: pip install arch")
        self.distribution = distribution
    
    def calculate(
        self,
        returns_df: pd.DataFrame,
        portfolio_weights: Optional[np.ndarray] = None,
        use_rolling_estimation: bool = True,
        window: int = 252
    ) -> dict:
        """
        Calculate GARCH volatility forecasts.
        
        Args:
            returns_df: DataFrame with returns (Date x Symbol)
            portfolio_weights: Portfolio weights (if None, forecasts per asset)
            use_rolling_estimation: Whether to use rolling window estimation
            window: Rolling window size
            
        Returns:
            Dictionary with volatility forecasts, runtime stats, and diagnostics
        """
        start_time = time.time()
        
        if portfolio_weights is not None:
            # Calculate portfolio returns
            portfolio_returns = np.dot(returns_df.values, portfolio_weights)
            returns_series = pd.Series(portfolio_returns, index=returns_df.index)
            symbols = ['Portfolio']
        else:
            returns_series = returns_df.iloc[:, 0]  # Use first asset for now
            symbols = [returns_df.columns[0]]
        
        # Fit GARCH(1,1) model
        try:
            model = arch_model(returns_series * 100, vol='Garch', p=1, q=1, dist=self.distribution)
            fitted_model = model.fit(disp='off')
            
            # Forecast volatility
            forecast = fitted_model.forecast(horizon=1)
            vol_forecast = forecast.variance.values[-1, 0] ** 0.5 / 100  # Convert back from percentage
            
            # Get fitted volatility
            fitted_vol = fitted_model.conditional_volatility / 100
            
        except Exception as e:
            print(f"GARCH fitting failed: {e}")
            # Fallback: use simple volatility
            vol_forecast = returns_series.std() * np.sqrt(252)
            fitted_vol = returns_series.std() * np.sqrt(252)
            fitted_model = None
        
        runtime = time.time() - start_time
        
        return {
            'asset_vol_forecasts': {symbols[0]: vol_forecast},
            'portfolio_vol_forecasts': vol_forecast if portfolio_weights is not None else None,
            'runtime_stats': {
                'wall_clock_time': runtime,
                'method': 'garch_11'
            },
            'diagnostics': {
                'fitted_volatility_mean': np.mean(fitted_vol),
                'fitted_volatility_std': np.std(fitted_vol),
                'aic': fitted_model.aic if fitted_model is not None else None,
                'bic': fitted_model.bic if fitted_model is not None else None
            }
        }

