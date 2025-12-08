"""Portfolio generation using Dirichlet distribution."""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from config.config_loader import get_config


class PortfolioGenerator:
    """Generates random portfolios using Dirichlet distribution."""
    
    def __init__(self):
        """Initialize the portfolio generator."""
        self.config = get_config()
        portfolio_config = self.config.get_portfolio_config()
        self.num_portfolios = portfolio_config.get('num_portfolios', 100000)
        self.constraints = portfolio_config.get('constraints', {})
        self.weight_dist = portfolio_config.get('weight_distribution', {})
        self.output_config = portfolio_config.get('output', {})
    
    def generate_dirichlet_portfolios(
        self, 
        num_assets: int, 
        num_portfolios: Optional[int] = None,
        alpha: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate portfolios using Dirichlet distribution.
        
        Args:
            num_assets: Number of assets
            num_portfolios: Number of portfolios to generate
            alpha: Dirichlet alpha parameter (if None, uses config)
            
        Returns:
            Array of shape (num_portfolios, num_assets) with portfolio weights
        """
        if num_portfolios is None:
            num_portfolios = self.num_portfolios
        
        if alpha is None:
            alpha = self.weight_dist.get('alpha_values', 0.5)
        
        # Generate Dirichlet samples
        # Dirichlet with alpha < 1 creates sparse/concentrated portfolios
        portfolios = np.random.dirichlet([alpha] * num_assets, size=num_portfolios)
        
        # Ensure constraints are met
        if self.constraints.get('long_only', True):
            portfolios = np.maximum(portfolios, 0)
        
        if self.constraints.get('fully_invested', True):
            # Normalize to sum to 1
            portfolios = portfolios / portfolios.sum(axis=1, keepdims=True)
        
        return portfolios
    
    def generate_and_save(
        self, 
        num_assets: int,
        symbols: list,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate portfolios and save to disk.
        
        Args:
            num_assets: Number of assets
            symbols: List of asset symbols
            output_path: Path to save portfolios (if None, uses config)
            
        Returns:
            DataFrame with portfolios (rows) and weights per asset (columns)
        """
        # Generate portfolios
        portfolios = self.generate_dirichlet_portfolios(num_assets)
        
        # Create DataFrame
        portfolios_df = pd.DataFrame(
            portfolios,
            columns=symbols
        )
        
        # Save to disk if configured
        if self.output_config.get('store_to_disk', True):
            if output_path is None:
                output_path = self.output_config.get('path', 'data/generated/portfolios_dirichlet.parquet')
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if output_path.suffix == '.parquet':
                portfolios_df.to_parquet(output_path, index=False)
            else:
                portfolios_df.to_csv(output_path, index=False)
            
            print(f"Saved {len(portfolios_df)} portfolios to {output_path}")
        
        return portfolios_df

