"""
Main evaluation script for Markowitz Mean-Variance Portfolio Optimization.

Orchestrates the entire optimization pipeline including:
- Data loading
- Returns computation
- Covariance and expected returns estimation
- Portfolio optimization
- Efficient frontier generation
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

# Suppress RuntimeWarning about module import in multiprocessing workers
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*found in sys.modules.*')

from .returns import (
    load_panel_prices,
    load_portfolio_universe,
    compute_daily_returns
)
from .markowitz_optimizer import (
    compute_covariance_matrix,
    compute_expected_returns,
    optimize_portfolio,
    generate_efficient_frontier
)
from .metrics import (
    compute_portfolio_statistics,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_max_drawdown,
    compute_calmar_ratio,
    compute_structure_metrics,
    compute_risk_metrics,
    compute_distribution_metrics,
    compute_runtime_metrics
)
from .report_generator import generate_report
from .time_sliced_metrics import compute_time_sliced_metrics


def _save_restructured_json(
    json_path: Path,
    portfolio_results: List[Dict],
    summary_stats: Dict,
    data_period: str,
    num_portfolios: int,
    markowitz_settings: Dict
):
    """
    Save results in the restructured JSON format.
    
    Args:
        json_path: Path to save JSON file
        portfolio_results: List of restructured portfolio results
        summary_stats: Summary statistics dictionary
        data_period: Data period string
        num_portfolios: Number of portfolios optimized
        markowitz_settings: Markowitz optimization settings
    """
    # Convert NaN to None for JSON serialization
    def clean_nan(obj):
        if isinstance(obj, dict):
            return {k: clean_nan(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_nan(item) for item in obj]
        elif isinstance(obj, (np.floating, float)) and np.isnan(obj):
            return None
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    output_data = {
        'metadata': {
            'task': 'markowitz_mean_variance_optimization',
            'data_period': data_period,
            'portfolios_optimized': num_portfolios,
            'markowitz_settings': markowitz_settings,
            'generated_at': datetime.now().isoformat()
        },
        'portfolio_results': clean_nan(portfolio_results),
        'summary': clean_nan(summary_stats)
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)


def _compute_summary_statistics(
    all_results: List[Dict],
    runtimes: List[float]
) -> Dict:
    """
    Compute summary statistics across all portfolios.
    
    Args:
        all_results: List of flat result dictionaries
        runtimes: List of runtime values
        
    Returns:
        Dictionary with summary statistics
    """
    if len(all_results) == 0:
        return {}
    
    results_df = pd.DataFrame(all_results)
    
    # Portfolio-level insights
    portfolio_insights = {}
    
    if 'expected_return' in results_df.columns:
        portfolio_insights['avg_expected_return'] = float(results_df['expected_return'].mean())
        portfolio_insights['max_expected_return'] = float(results_df['expected_return'].max())
    
    if 'volatility' in results_df.columns:
        portfolio_insights['avg_volatility'] = float(results_df['volatility'].mean())
        portfolio_insights['min_volatility'] = float(results_df['volatility'].min())
    
    if 'sharpe_ratio' in results_df.columns:
        portfolio_insights['avg_sharpe_ratio'] = float(results_df['sharpe_ratio'].mean())
        portfolio_insights['max_sharpe_ratio'] = float(results_df['sharpe_ratio'].max())
    
    # Distribution effects
    distribution_effects = {}
    if 'skewness' in results_df.columns:
        distribution_effects['avg_skew'] = float(results_df['skewness'].mean())
    if 'kurtosis' in results_df.columns:
        distribution_effects['avg_kurtosis'] = float(results_df['kurtosis'].mean())
    if 'jarque_bera_p_value' in results_df.columns:
        rejection_rate = (results_df['jarque_bera_p_value'] < 0.05).mean()
        distribution_effects['normality_rejection_rate'] = float(rejection_rate)
    
    # Structure effects
    structure_effects = {}
    if 'hhi_concentration' in results_df.columns and 'sharpe_ratio' in results_df.columns:
        corr = results_df['hhi_concentration'].corr(results_df['sharpe_ratio'])
        if not np.isnan(corr):
            structure_effects['correlation_hhi_vs_sharpe'] = float(corr)
    
    if 'effective_number_of_assets' in results_df.columns and 'volatility' in results_df.columns:
        corr = results_df['effective_number_of_assets'].corr(results_df['volatility'])
        if not np.isnan(corr):
            structure_effects['correlation_enc_vs_volatility'] = float(corr)
    
    # Runtime stats
    runtime_stats = {}
    if len(runtimes) > 0:
        runtime_array = np.array(runtimes) * 1000  # Convert to ms
        runtime_stats['mean_runtime_ms'] = float(np.mean(runtime_array))
        runtime_stats['p95_runtime_ms'] = float(np.percentile(runtime_array, 95))
        runtime_stats['median_runtime_ms'] = float(np.median(runtime_array))
    
    return {
        'portfolio_level_insights': portfolio_insights,
        'distribution_effects': distribution_effects,
        'structure_effects': structure_effects,
        'runtime_stats': runtime_stats
    }


def _process_single_optimization(
    returns: pd.DataFrame,
    portfolio_universe: pd.DataFrame,
    portfolio_id: Union[int, str],
    markowitz_settings: Dict,
    risk_free_rate: float,
    random_seed: Optional[int] = None
) -> Tuple[Dict, float]:
    """
    Process a single portfolio optimization.
    
    Args:
        returns: DataFrame of daily returns
        portfolio_universe: DataFrame with portfolio universe information
        portfolio_id: Portfolio identifier
        markowitz_settings: Markowitz optimization settings
        risk_free_rate: Risk-free rate
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (result dictionary, runtime in seconds)
    """
    start_time = time.time()
    
    try:
        # Get assets for this portfolio
        if portfolio_id in portfolio_universe.index:
            portfolio_assets = portfolio_universe.loc[portfolio_id]
            # Handle different formats
            if isinstance(portfolio_assets, pd.Series):
                # Get assets with non-zero weights or all assets if weights not available
                if 'weight' in portfolio_assets.index or 'weights' in portfolio_assets.index:
                    # Has weights
                    pass
                assets = portfolio_assets.index.tolist()
            else:
                assets = portfolio_assets
        else:
            # Use all available assets
            assets = returns.columns.tolist()
        
        # Filter to assets available in returns
        assets = [a for a in assets if a in returns.columns]
        
        if len(assets) == 0:
            return {}, 0.0
        
        returns_subset = returns[assets]
        
        # Get settings
        cov_settings = markowitz_settings.get('covariance_estimation', {})
        exp_return_settings = markowitz_settings.get('expected_return_estimation', {})
        constraints = markowitz_settings.get('constraints', {})
        solver_settings = markowitz_settings.get('optimization_solver', {})
        
        # Compute covariance matrix
        cov_matrix, cov_time = compute_covariance_matrix(
            returns_subset,
            method=cov_settings.get('method', 'sample'),
            window=cov_settings.get('estimation_windows', [252])[0] if cov_settings.get('estimation_windows') else None,
            use_shrinkage=cov_settings.get('shrinkage', {}).get('use_shrinkage', False),
            shrinkage_method=cov_settings.get('shrinkage', {}).get('method', 'ledoit_wolf')
        )
        
        # Compute expected returns
        expected_returns = compute_expected_returns(
            returns_subset,
            method=exp_return_settings.get('method', 'historical_mean'),
            window=exp_return_settings.get('estimation_windows', [252])[0] if exp_return_settings.get('estimation_windows') else None,
            use_annualization=exp_return_settings.get('use_annualization', True)
        )
        
        # Determine objective and risk aversion
        objective = markowitz_settings.get('objective', 'min_variance')
        risk_aversion = None
        
        if markowitz_settings.get('risk_return_tradeoff', {}).get('use_risk_aversion', False):
            lambda_values = markowitz_settings['risk_return_tradeoff'].get('lambda_values', [1.0])
            risk_aversion = lambda_values[0]  # Use first lambda value
        
        # Optimize portfolio
        optimal_weights, opt_info, solver_time = optimize_portfolio(
            expected_returns,
            cov_matrix,
            objective=objective,
            risk_aversion=risk_aversion,
            risk_free_rate=risk_free_rate,
            constraints=constraints,
            solver=solver_settings.get('qp_backend', 'osqp'),
            tolerance=solver_settings.get('tolerance', 1e-6)
        )
        
        # Compute portfolio returns
        portfolio_returns = (returns_subset * optimal_weights).sum(axis=1)
        
        # Compute all metrics
        portfolio_stats = compute_portfolio_statistics(
            portfolio_returns,
            optimal_weights,
            expected_returns,
            cov_matrix,
            risk_free_rate
        )
        
        sharpe = compute_sharpe_ratio(portfolio_returns, risk_free_rate)
        sortino = compute_sortino_ratio(portfolio_returns, risk_free_rate)
        max_dd = compute_max_drawdown(portfolio_returns)
        calmar = compute_calmar_ratio(portfolio_returns)
        
        structure_metrics = compute_structure_metrics(
            optimal_weights,
            cov_matrix,
            returns_subset
        )
        
        risk_metrics = compute_risk_metrics(portfolio_returns)
        
        distribution_metrics = compute_distribution_metrics(portfolio_returns)
        
        runtime_total = time.time() - start_time
        
        # Combine all results
        result = {
            'portfolio_id': portfolio_id,
            'objective': objective,
            'risk_aversion': risk_aversion,
            **portfolio_stats,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'calmar_ratio': calmar,
            **structure_metrics,
            **risk_metrics,
            **distribution_metrics,
            'runtime_per_optimization_ms': runtime_total * 1000,
            'covariance_estimation_time_ms': cov_time,
            'solver_time_ms': solver_time,
            'optimization_status': opt_info.get('status', 'unknown')
        }
        
        return result, runtime_total
        
    except Exception as e:
        warnings.warn(f"Error optimizing portfolio {portfolio_id}: {e}")
        return {}, 0.0


def run_markowitz_optimization(
    config_path: Optional[Union[str, Path]] = None,
    config_dict: Optional[Dict] = None,
    max_portfolios: Optional[int] = 100
) -> Dict:
    """
    Run Markowitz mean-variance optimization pipeline.
    
    Args:
        config_path: Path to llm.json configuration file
        config_dict: Optional configuration dictionary (overrides config_path)
        max_portfolios: Maximum number of portfolios to process (default: 100, None for all)
        
    Returns:
        Dictionary with optimization results
    """
    # Load configuration
    if config_dict is None:
        if config_path is None:
            # Default path
            current_file = Path(__file__)
            config_path = current_file.parent / 'llm.json'
        
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = config_dict
    
    # Extract settings
    inputs = config['inputs']
    markowitz_settings = config['markowitz_settings']
    modules = config.get('modules', {})
    outputs = config['outputs']
    report_sections = config.get('report_sections', [])
    
    print("Loading data...")
    # Load data
    prices = load_panel_prices(inputs['panel_price_path'])
    portfolio_universe = load_portfolio_universe(inputs['portfolio_universe_path'])
    risk_free_rate = inputs.get('risk_free_rate', 0.0)
    
    print("Computing returns...")
    # Compute returns
    daily_returns = compute_daily_returns(prices, method=inputs.get('return_type', 'log'))
    
    # Get data period
    data_period = f"{prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}"
    
    # Get portfolio IDs
    if isinstance(portfolio_universe, pd.DataFrame):
        portfolio_ids = portfolio_universe.index.tolist()
    else:
        portfolio_ids = [0]  # Single portfolio case
    
    num_portfolios_total = len(portfolio_ids)
    
    # Limit number of portfolios if specified
    if max_portfolios is not None and max_portfolios > 0:
        num_portfolios = min(num_portfolios_total, max_portfolios)
        portfolio_ids = portfolio_ids[:num_portfolios]
        print(f"Optimizing {num_portfolios:,} portfolios (out of {num_portfolios_total:,} total)...")
    else:
        num_portfolios = num_portfolios_total
        print(f"Optimizing {num_portfolios:,} portfolios...")
    
    # Process portfolios
    all_results = []
    all_runtimes = []
    all_weights = []
    efficient_frontiers = []
    
    for portfolio_id in portfolio_ids:
        result, runtime = _process_single_optimization(
            daily_returns,
            portfolio_universe,
            portfolio_id,
            markowitz_settings,
            risk_free_rate,
            random_seed=markowitz_settings.get('random_seed')
        )
        
        if result:
            all_results.append(result)
            all_runtimes.append(runtime)
            
            # Store weights (would need to be retrieved from optimization)
            # For now, we'll compute it in the results
    
    if len(all_results) == 0:
        raise RuntimeError("No portfolios were successfully optimized")
    
    print("Computing summary statistics...")
    # Compute summary statistics
    summary_stats = _compute_summary_statistics(all_results, all_runtimes)
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame(all_results)
    
    # Generate efficient frontier if requested
    if modules.get('generate_efficient_frontier', False):
        print("Generating efficient frontier...")
        try:
            # Use first portfolio's assets for frontier
            assets = daily_returns.columns.tolist()
            returns_subset = daily_returns[assets]
            
            cov_settings = markowitz_settings.get('covariance_estimation', {})
            cov_matrix, _ = compute_covariance_matrix(
                returns_subset,
                method=cov_settings.get('method', 'sample'),
                use_shrinkage=cov_settings.get('shrinkage', {}).get('use_shrinkage', False),
                shrinkage_method=cov_settings.get('shrinkage', {}).get('method', 'ledoit_wolf')
            )
            
            expected_returns = compute_expected_returns(
                returns_subset,
                method=markowitz_settings.get('expected_return_estimation', {}).get('method', 'historical_mean'),
                use_annualization=markowitz_settings.get('expected_return_estimation', {}).get('use_annualization', True)
            )
            
            frontier = generate_efficient_frontier(
                expected_returns,
                cov_matrix,
                num_portfolios=markowitz_settings.get('frontier_settings', {}).get('num_portfolios', 100),
                risk_levels=markowitz_settings.get('frontier_settings', {}).get('risk_levels', 'auto'),
                constraints=markowitz_settings.get('constraints', {}),
                risk_free_rate=risk_free_rate,
                solver=markowitz_settings.get('optimization_solver', {}).get('qp_backend', 'osqp')
            )
            efficient_frontiers.append(frontier)
        except Exception as e:
            warnings.warn(f"Failed to generate efficient frontier: {e}")
    
    print("Saving results...")
    # Save outputs
    output_base = Path(outputs['metrics_table']).parent
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Save metrics table
    metrics_df.to_parquet(outputs['metrics_table'])
    
    # Save optimal portfolios (weights)
    if 'optimal_portfolios' in outputs:
        # Create weights DataFrame (would need to be stored during optimization)
        # For now, create placeholder
        weights_df = pd.DataFrame(index=metrics_df['portfolio_id'])
        weights_df.to_parquet(outputs['optimal_portfolios'])
    
    # Save efficient frontier
    if 'efficient_frontier' in outputs and len(efficient_frontiers) > 0:
        efficient_frontiers[0].to_parquet(outputs['efficient_frontier'])
    
    # Save JSON
    portfolio_results = []
    for _, row in metrics_df.iterrows():
        portfolio_results.append(row.to_dict())
    
    _save_restructured_json(
        Path(outputs['metrics_json']),
        portfolio_results,
        summary_stats,
        data_period,
        num_portfolios,
        markowitz_settings
    )
    
    print("Generating report...")
    # Generate report
    generate_report(
        metrics_df,
        outputs['summary_report'],
        markowitz_settings,
        report_sections
    )
    
    print("Optimization complete!")
    
    return {
        'metrics_df': metrics_df,
        'summary_stats': summary_stats,
        'efficient_frontiers': efficient_frontiers,
        'num_portfolios': num_portfolios,
        'num_portfolios_total': num_portfolios_total
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run Markowitz Mean-Variance Portfolio Optimization'
    )
    parser.add_argument(
        '--max-portfolios',
        type=int,
        default=100,
        help='Maximum number of portfolios to process (default: 100, use 0 for all)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration JSON file (default: llm.json in module directory)'
    )
    
    args = parser.parse_args()
    
    # Convert 0 to None (process all portfolios)
    max_portfolios = None if args.max_portfolios == 0 else args.max_portfolios
    
    # Run optimization
    results = run_markowitz_optimization(
        config_path=args.config,
        max_portfolios=max_portfolios
    )
    print(f"Optimized {results['num_portfolios']:,} portfolios (out of {results['num_portfolios_total']:,} total)")

