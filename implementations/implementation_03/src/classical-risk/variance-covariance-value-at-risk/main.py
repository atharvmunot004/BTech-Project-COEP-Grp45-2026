"""
Main evaluation script for Variance-Covariance VaR.

Orchestrates the entire VaR evaluation pipeline including:
- Data loading
- Returns computation
- Rolling VaR calculation
- Backtesting
- Metrics computation
- Report generation
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings

from .returns import (
    load_panel_prices,
    load_portfolio_weights,
    compute_daily_returns,
    compute_portfolio_returns
)
from .var_calculator import compute_rolling_var, align_returns_and_var
from .backtesting import compute_accuracy_metrics
from .metrics import (
    compute_tail_metrics,
    compute_structure_metrics,
    compute_distribution_metrics,
    compute_runtime_metrics
)
from .report_generator import generate_report


def _process_single_portfolio(
    portfolio_data: Tuple[int, Tuple, pd.Series],
    daily_returns: pd.DataFrame,
    confidence_levels: List[float],
    horizons: List[int],
    estimation_windows: List[int]
) -> Tuple[List[Dict], float]:
    """
    Process a single portfolio and return all results for all configurations.
    
    This is a worker function designed to be called in parallel.
    
    Args:
        portfolio_data: Tuple of (portfolio_idx, portfolio_id, portfolio_weights)
        daily_returns: DataFrame of daily returns
        confidence_levels: List of confidence levels
        horizons: List of horizons
        estimation_windows: List of estimation windows
        
    Returns:
        Tuple of (list of result dictionaries, runtime in seconds)
    """
    portfolio_idx, portfolio_id, portfolio_weights = portfolio_data
    
    start_time = time.time()
    results = []
    
    try:
        # Compute portfolio returns
        portfolio_returns = compute_portfolio_returns(
            daily_returns,
            portfolio_weights,
            align_assets=True
        )
    except Exception as e:
        # Return empty list on error
        return results
    
    # Compute covariance matrix for structure metrics (once per portfolio)
    try:
        common_assets = daily_returns.columns.intersection(portfolio_weights.index)
        returns_aligned = daily_returns[common_assets]
        covariance_matrix = returns_aligned.cov()
    except:
        covariance_matrix = None
    
    # Evaluate for each combination of settings
    for confidence_level in confidence_levels:
        for horizon in horizons:
            for window in estimation_windows:
                try:
                    # Compute rolling VaR
                    rolling_var = compute_rolling_var(
                        daily_returns,
                        portfolio_weights,
                        window=window,
                        confidence_level=confidence_level,
                        horizon=horizon
                    )
                    
                    # Align returns and VaR
                    aligned_returns, aligned_var = align_returns_and_var(
                        portfolio_returns,
                        rolling_var
                    )
                    
                    if len(aligned_returns) == 0:
                        continue
                    
                    # Compute accuracy metrics
                    accuracy_metrics = compute_accuracy_metrics(
                        aligned_returns,
                        aligned_var,
                        confidence_level=confidence_level
                    )
                    
                    # Compute tail metrics
                    tail_metrics = compute_tail_metrics(
                        aligned_returns,
                        aligned_var
                    )
                    
                    # Compute structure metrics
                    structure_metrics = compute_structure_metrics(
                        portfolio_weights,
                        covariance_matrix
                    )
                    
                    # Compute distribution metrics
                    distribution_metrics = compute_distribution_metrics(
                        aligned_returns
                    )
                    
                    # Combine all metrics
                    result = {
                        'portfolio_id': portfolio_id,
                        'confidence_level': confidence_level,
                        'horizon': horizon,
                        'estimation_window': window,
                        **accuracy_metrics,
                        **tail_metrics,
                        **structure_metrics,
                        **distribution_metrics
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    # Continue to next configuration on error
                    continue
    
    runtime = time.time() - start_time
    return results, runtime


def evaluate_var(
    config_path: Optional[Union[str, Path]] = None,
    config_dict: Optional[Dict] = None,
    n_jobs: Optional[int] = None
) -> pd.DataFrame:
    """
    Main function to evaluate VaR for multiple portfolios.
    
    Args:
        config_path: Path to JSON configuration file
        config_dict: Configuration dictionary (if not loading from file)
        n_jobs: Number of parallel workers (default: number of CPU cores)
        
    Returns:
        DataFrame with all computed metrics
    """
    # Load configuration
    if config_dict is None:
        if config_path is None:
            # Default to llm.json in same directory
            config_path = Path(__file__).parent / "llm.json"
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = config_dict
    
    # Get paths (resolve relative to implementation_03 root)
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent.parent
    
    panel_price_path = project_root / config['inputs']['panel_price_path']
    portfolio_weights_path = project_root / config['inputs']['portfolio_weights_path']
    
    # Adjust path if it says "preprocessed" but file is in "processed"
    if not panel_price_path.exists() and 'preprocessed' in str(panel_price_path):
        panel_price_path = project_root / str(config['inputs']['panel_price_path']).replace('preprocessed', 'processed')
    
    print("=" * 80)
    print("VARIANCE-COVARIANCE VALUE-AT-RISK EVALUATION")
    print("=" * 80)
    print(f"\nLoading data...")
    print(f"  Panel prices: {panel_price_path}")
    print(f"  Portfolio weights: {portfolio_weights_path}")
    
    # Load data
    prices = load_panel_prices(panel_price_path)
    portfolio_weights_df = load_portfolio_weights(portfolio_weights_path)
    
    print(f"\nLoaded:")
    print(f"  Prices: {len(prices)} dates, {len(prices.columns)} assets")
    print(f"  Portfolios: {len(portfolio_weights_df)} portfolios")
    
    # Compute daily returns
    print(f"\nComputing daily returns...")
    daily_returns = compute_daily_returns(prices, method='log')
    print(f"  Daily returns: {len(daily_returns)} dates")
    
    # Get VaR settings
    confidence_levels = config['var_settings']['confidence_levels']
    horizons = config['var_settings']['horizons']
    estimation_windows = config['var_settings']['estimation_windows']
    
    # Initialize results list
    all_results = []
    runtimes = []
    
    # Process each portfolio
    print(f"\nEvaluating {len(portfolio_weights_df)} portfolios...")
    print(f"  Confidence levels: {confidence_levels}")
    print(f"  Horizons: {horizons} days")
    print(f"  Estimation windows: {estimation_windows} days")
    
    num_portfolios = len(portfolio_weights_df)
    
    # Calculate total combinations for progress tracking
    total_combinations = num_portfolios * len(confidence_levels) * len(horizons) * len(estimation_windows)
    print(f"  Total portfolio-configuration combinations: {total_combinations:,}")
    
    # Determine number of workers
    if n_jobs is None:
        n_jobs = cpu_count()
    elif n_jobs == -1:
        n_jobs = cpu_count()
    elif n_jobs < 1:
        n_jobs = 1
    
    print(f"  Using {n_jobs} parallel workers...")
    print(f"  Processing all {num_portfolios:,} portfolios...")
    
    # Prepare portfolio data for parallel processing
    portfolio_data_list = [
        (idx, portfolio_id, portfolio_weights)
        for idx, (portfolio_id, portfolio_weights) in enumerate(portfolio_weights_df.iterrows())
    ]
    
    # Create worker function with fixed arguments
    worker_func = partial(
        _process_single_portfolio,
        daily_returns=daily_returns,
        confidence_levels=confidence_levels,
        horizons=horizons,
        estimation_windows=estimation_windows
    )
    
    # Process portfolios in parallel
    start_time_total = time.time()
    
    if n_jobs == 1:
        # Sequential processing (useful for debugging)
        print("  Running in sequential mode...")
        for portfolio_idx, portfolio_data in enumerate(portfolio_data_list):
            if (portfolio_idx + 1) % 100 == 0 or (portfolio_idx + 1) in [1, 10, 50, 500, 1000, 5000, 10000]:
                print(f"  Processing portfolio {portfolio_idx + 1:,}/{num_portfolios:,} ({100*(portfolio_idx+1)/num_portfolios:.1f}%)...")
            
            results, runtime = worker_func(portfolio_data)
            all_results.extend(results)
            runtimes.append(runtime)
    else:
        # Parallel processing
        print(f"  Running in parallel mode with {n_jobs} workers...")
        completed = 0
        
        with Pool(processes=n_jobs) as pool:
            # Use imap for progress tracking
            results_iter = pool.imap(worker_func, portfolio_data_list, chunksize=max(1, num_portfolios // (n_jobs * 4)))
            
            for results, runtime in results_iter:
                completed += 1
                all_results.extend(results)
                runtimes.append(runtime)
                
                # Progress reporting
                if completed % 100 == 0 or completed in [1, 10, 50, 500, 1000, 5000, 10000]:
                    elapsed = time.time() - start_time_total
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = (num_portfolios - completed) / rate if rate > 0 else 0
                    print(f"  Completed {completed:,}/{num_portfolios:,} ({100*completed/num_portfolios:.1f}%) | "
                          f"Rate: {rate:.1f} portfolios/sec | "
                          f"Elapsed: {elapsed/60:.1f} min | "
                          f"Remaining: {remaining/60:.1f} min")
    
    total_runtime = time.time() - start_time_total
    avg_runtime_per_portfolio = total_runtime / num_portfolios if num_portfolios > 0 else 0
    
    # Create results DataFrame
    if len(all_results) == 0:
        raise ValueError("No results computed. Check data and configuration.")
    
    results_df = pd.DataFrame(all_results)
    
    # Add runtime metrics
    runtime_metrics = compute_runtime_metrics(runtimes)
    for key, value in runtime_metrics.items():
        results_df[key] = value
    
    print(f"\nCompleted evaluation of {len(results_df)} portfolio-configuration combinations")
    print(f"  Total runtime: {total_runtime/60:.2f} minutes ({total_runtime:.2f} seconds)")
    print(f"  Average runtime: {avg_runtime_per_portfolio*1000:.2f} ms per portfolio")
    if n_jobs > 1:
        print(f"  Speedup: ~{n_jobs}x (theoretical maximum with {n_jobs} workers)")
    
    # Save results
    outputs = config.get('outputs', {})
    
    if 'metrics_table' in outputs:
        metrics_path = project_root / outputs['metrics_table']
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving metrics table...")
        print(f"  Path: {metrics_path}")
        
        if metrics_path.suffix == '.parquet':
            results_df.to_parquet(metrics_path, index=False)
        elif metrics_path.suffix == '.csv':
            results_df.to_csv(metrics_path, index=False)
        else:
            # Default to parquet
            metrics_path = metrics_path.with_suffix('.parquet')
            results_df.to_parquet(metrics_path, index=False)
        
        print(f"  Saved: {metrics_path}")
    
    # Generate report
    if 'summary_report' in outputs:
        report_path = project_root / outputs['summary_report']
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating report...")
        print(f"  Path: {report_path}")
        
        generate_report(
            results_df,
            report_path,
            var_settings=config.get('var_settings'),
            report_sections=config.get('report_sections')
        )
        
        print(f"  Saved: {report_path}")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    
    return results_df


def main():
    """Main entry point for command-line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Evaluate Variance-Covariance VaR for portfolios'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration JSON file (default: llm.json in same directory)'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=None,
        help='Number of parallel workers (default: number of CPU cores, use -1 for all cores, 1 for sequential)'
    )
    
    args = parser.parse_args()
    
    # Handle -1 for all cores
    n_jobs = args.n_jobs
    if n_jobs == -1:
        n_jobs = None
    
    results_df = evaluate_var(config_path=args.config, n_jobs=n_jobs)
    
    print(f"\nResults summary:")
    print(f"  Total rows: {len(results_df)}")
    print(f"  Columns: {len(results_df.columns)}")
    print(f"\nFirst few rows:")
    print(results_df.head())


if __name__ == "__main__":
    main()

