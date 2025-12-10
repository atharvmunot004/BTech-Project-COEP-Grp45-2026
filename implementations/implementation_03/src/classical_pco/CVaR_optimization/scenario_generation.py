"""
Scenario generation module for CVaR optimization.

Generates scenario matrices from historical returns for CVaR linear programming.
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple
import time
import warnings


def generate_scenario_matrix(
    returns: pd.DataFrame,
    source: str = 'historical',
    estimation_window: Optional[int] = None,
    num_scenarios: Optional[int] = None,
    use_block_bootstrap: bool = False,
    block_length: Optional[int] = None,
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, float]:
    """
    Generate scenario matrix for CVaR optimization.
    
    The scenario matrix has shape (num_scenarios, num_assets) where each row
    represents a possible future return scenario.
    
    Args:
        returns: DataFrame of historical returns with dates as index and assets as columns
        source: 'historical' to use historical returns as scenarios
        estimation_window: Number of historical periods to use (if None, uses all)
        num_scenarios: Number of scenarios to generate (if None, uses all available)
        use_block_bootstrap: Whether to use block bootstrap for scenario generation
        block_length: Block length for block bootstrap (if None, auto-determined)
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (scenario_matrix, computation_time_ms)
    """
    start_time = time.time()
    
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Select data window
    if estimation_window is not None and len(returns) > estimation_window:
        returns_window = returns.iloc[-estimation_window:]
    else:
        returns_window = returns
    
    # Remove any assets with insufficient data
    returns_window = returns_window.dropna(axis=1, how='any')
    
    if len(returns_window.columns) == 0:
        raise ValueError("No assets with sufficient data for scenario generation")
    
    if len(returns_window) == 0:
        raise ValueError("No historical data available for scenario generation")
    
    # Generate scenarios
    if source == 'historical':
        if use_block_bootstrap:
            scenario_matrix = _block_bootstrap_scenarios(
                returns_window,
                num_scenarios=num_scenarios,
                block_length=block_length,
                random_seed=random_seed
            )
        else:
            # Use historical returns directly as scenarios
            scenario_matrix = returns_window.values
            
            # If num_scenarios is specified and less than available, sample randomly
            if num_scenarios is not None and num_scenarios < len(scenario_matrix):
                indices = np.random.choice(
                    len(scenario_matrix),
                    size=num_scenarios,
                    replace=False
                )
                scenario_matrix = scenario_matrix[indices]
    else:
        raise ValueError(f"Unknown scenario source: {source}")
    
    computation_time = (time.time() - start_time) * 1000  # Convert to ms
    
    return scenario_matrix, computation_time


def _block_bootstrap_scenarios(
    returns: pd.DataFrame,
    num_scenarios: Optional[int] = None,
    block_length: Optional[int] = None,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate scenarios using block bootstrap method.
    
    Args:
        returns: DataFrame of historical returns
        num_scenarios: Number of scenarios to generate
        block_length: Block length (if None, uses optimal block length)
        random_seed: Random seed
        
    Returns:
        Scenario matrix
    """
    n_periods, n_assets = returns.shape
    
    # Determine block length if not provided
    if block_length is None:
        # Use rule of thumb: sqrt(n) or 20, whichever is smaller
        block_length = min(int(np.sqrt(n_periods)), 20)
        block_length = max(block_length, 1)
    
    # Determine number of scenarios
    if num_scenarios is None:
        num_scenarios = n_periods
    
    # Generate scenarios
    scenarios = []
    n_blocks_needed = int(np.ceil(num_scenarios / block_length))
    
    for _ in range(n_blocks_needed):
        # Randomly select a starting point
        max_start = n_periods - block_length + 1
        if max_start <= 0:
            # If block length >= n_periods, use all data
            block = returns.values
        else:
            start_idx = np.random.randint(0, max_start)
            block = returns.iloc[start_idx:start_idx + block_length].values
        
        scenarios.append(block)
    
    # Concatenate and trim to desired number
    scenario_matrix = np.vstack(scenarios)[:num_scenarios]
    
    return scenario_matrix


def compute_portfolio_scenario_returns(
    scenario_matrix: np.ndarray,
    weights: np.ndarray
) -> np.ndarray:
    """
    Compute portfolio returns for each scenario.
    
    Args:
        scenario_matrix: Scenario matrix of shape (num_scenarios, num_assets)
        weights: Portfolio weights of shape (num_assets,)
        
    Returns:
        Array of portfolio returns for each scenario
    """
    # Portfolio return = sum(weight_i * return_i) for each scenario
    portfolio_returns = scenario_matrix @ weights
    
    return portfolio_returns

