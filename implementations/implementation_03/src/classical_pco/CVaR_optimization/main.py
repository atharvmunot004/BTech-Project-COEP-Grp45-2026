"""
Main evaluation script for CVaR Portfolio Optimization.

Orchestrates the entire optimization pipeline including:
- Data loading
- Returns computation
- Scenario matrix generation
- CVaR optimization using Rockafellar-Uryasev formulation
- CVaR-return frontier generation
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
import warnings

from .returns import (
    load_panel_prices,
    load_baseline_portfolios,
    compute_daily_returns
)
from .scenario_generation import (
    generate_scenario_matrix,
    compute_portfolio_scenario_returns
)
from .cvar_optimizer import (
    optimize_cvar_portfolio,
    generate_cvar_return_frontier
)
from .metrics import (
    compute_portfolio_statistics,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_max_drawdown,
    compute_calmar_ratio,
    compute_cvar_sharpe_ratio,
    compute_cvar_sortino_ratio,
    compute_cvar_calmar_ratio,
    compute_return_over_cvar,
    compute_return_over_var,
    compute_var_cvar,
    compute_risk_metrics,
    compute_cvar_sensitivity,
    compute_var_sensitivity,
    compute_structure_metrics,
    compute_distribution_metrics,
    compute_tail_risk_metrics,
    compute_runtime_metrics
)
from .report_generator import generate_report


def _save_restructured_json(
    json_path: Path,
    portfolio_results: List[Dict],
    summary_stats: Dict,
    data_period: str,
    num_portfolios: int,
    cvar_settings: Dict
):
    """
    Save results in the restructured JSON format.
    
    Args:
        json_path: Path to save JSON file
        portfolio_results: List of restructured portfolio results
        summary_stats: Summary statistics dictionary
        data_period: Data period string
        num_portfolios: Number of portfolios optimized
        cvar_settings: CVaR optimization settings
    """
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
            'task': 'cvar_optimization',
            'data_period': data_period,
            'portfolios_optimized': num_portfolios,
            'cvar_settings': cvar_settings,
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
    
    portfolio_insights = {}
    
    if 'expected_return' in results_df.columns:
        portfolio_insights['avg_expected_return'] = float(results_df['expected_return'].mean())
        portfolio_insights['max_expected_return'] = float(results_df['expected_return'].max())
    
    if 'cvar' in results_df.columns:
        portfolio_insights['avg_cvar'] = float(results_df['cvar'].mean())
        portfolio_insights['min_cvar'] = float(results_df['cvar'].min())
    
    if 'cvar_sharpe_ratio' in results_df.columns:
        portfolio_insights['avg_cvar_sharpe'] = float(results_df['cvar_sharpe_ratio'].mean())
        portfolio_insights['max_cvar_sharpe'] = float(results_df['cvar_sharpe_ratio'].max())
    
    runtime_stats = {}
    if len(runtimes) > 0:
        runtime_array = np.array(runtimes) * 1000
        runtime_stats['mean_runtime_ms'] = float(np.mean(runtime_array))
        runtime_stats['p95_runtime_ms'] = float(np.percentile(runtime_array, 95))
    
    return {
        'portfolio_level_insights': portfolio_insights,
        'runtime_stats': runtime_stats
    }


def _process_single_optimization(
    returns: pd.DataFrame,
    baseline_portfolios: pd.DataFrame,
    portfolio_id: Union[int, str],
    cvar_settings: Dict,
    risk_free_rate: float,
    random_seed: Optional[int] = None
) -> Tuple[Dict, float]:
    """
    Process a single portfolio CVaR optimization.
    
    Args:
        returns: DataFrame of daily returns
        baseline_portfolios: DataFrame with baseline portfolio information
        portfolio_id: Portfolio identifier
        cvar_settings: CVaR optimization settings
        risk_free_rate: Risk-free rate
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (result dictionary, runtime in seconds)
    """
    start_time = time.time()
    
    try:
        # Get assets for this portfolio
        if portfolio_id in baseline_portfolios.index:
            portfolio_assets = baseline_portfolios.loc[portfolio_id]
            if isinstance(portfolio_assets, pd.Series):
                assets = portfolio_assets.index.tolist()
            else:
                assets = portfolio_assets
        else:
            assets = returns.columns.tolist()
        
        # Filter to assets available in returns
        assets = [a for a in assets if a in returns.columns]
        
        if len(assets) == 0:
            return {}, 0.0
        
        returns_subset = returns[assets]
        
        # Get settings
        scenario_settings = cvar_settings.get('scenario_generation', {})
        constraints = cvar_settings.get('constraints', {})
        solver_settings = cvar_settings.get('optimization_solver', {})
        confidence_levels = cvar_settings.get('confidence_levels', [0.95])
        confidence_level = confidence_levels[0]  # Use first confidence level
        
        # Generate scenario matrix
        estimation_windows = scenario_settings.get('estimation_windows', [252])
        estimation_window = estimation_windows[0]  # Use first window
        
        scenario_matrix, scenario_time = generate_scenario_matrix(
            returns_subset,
            source=scenario_settings.get('source', 'historical'),
            estimation_window=estimation_window,
            num_scenarios=scenario_settings.get('num_scenarios'),
            use_block_bootstrap=scenario_settings.get('use_block_bootstrap', False),
            random_seed=random_seed
        )
        
        # Compute expected returns (mean of scenarios)
        expected_returns = pd.Series(
            scenario_matrix.mean(axis=0),
            index=returns_subset.columns
        )
        
        # Determine optimization parameters
        risk_return_tradeoff = cvar_settings.get('risk_return_tradeoff', {})
        lambda_cvar = risk_return_tradeoff.get('lambda_cvar', [1.0])[0]
        lambda_mean_return = risk_return_tradeoff.get('lambda_mean_return', [0.0])[0]
        target_return = None
        
        # Optimize portfolio
        optimal_weights, opt_info, solver_time = optimize_cvar_portfolio(
            scenario_matrix,
            expected_returns=expected_returns.values,
            confidence_level=confidence_level,
            target_return=target_return,
            lambda_cvar=lambda_cvar,
            lambda_mean_return=lambda_mean_return,
            constraints=constraints,
            solver=solver_settings.get('lp_backend', 'highs'),
            tolerance=solver_settings.get('tolerance', 1e-6),
            max_iterations=solver_settings.get('max_iterations', 100000)
        )
        
        # Compute portfolio returns
        portfolio_returns = (returns_subset * optimal_weights).sum(axis=1)
        
        # Compute all metrics
        portfolio_stats = compute_portfolio_statistics(
            portfolio_returns,
            optimal_weights,
            expected_returns,
            risk_free_rate
        )
        
        sharpe = compute_sharpe_ratio(portfolio_returns, risk_free_rate)
        sortino = compute_sortino_ratio(portfolio_returns, risk_free_rate)
        max_dd = compute_max_drawdown(portfolio_returns)
        calmar = compute_calmar_ratio(portfolio_returns)
        
        # CVaR-specific metrics
        cvar_sharpe = compute_cvar_sharpe_ratio(portfolio_returns, risk_free_rate, confidence_level)
        cvar_sortino = compute_cvar_sortino_ratio(portfolio_returns, risk_free_rate, confidence_level)
        cvar_calmar = compute_cvar_calmar_ratio(portfolio_returns, confidence_level)
        return_over_cvar_val = compute_return_over_cvar(portfolio_returns, confidence_level)
        return_over_var_val = compute_return_over_var(portfolio_returns, confidence_level)
        
        # Risk metrics
        var, cvar = compute_var_cvar(portfolio_returns, confidence_level)
        risk_metrics = compute_risk_metrics(portfolio_returns, confidence_level)
        
        # Structure metrics
        structure_metrics = compute_structure_metrics(
            optimal_weights,
            returns_subset
        )
        
        # Distribution metrics
        distribution_metrics = compute_distribution_metrics(portfolio_returns)
        
        # Tail risk metrics
        tail_risk_metrics = compute_tail_risk_metrics(
            portfolio_returns,
            confidence_levels=confidence_levels
        )
        
        # CVaR/VaR sensitivity
        cvar_sensitivity = compute_cvar_sensitivity(portfolio_returns, confidence_levels)
        var_sensitivity = compute_var_sensitivity(portfolio_returns, confidence_levels)
        
        runtime_total = time.time() - start_time
        
        # Combine all results
        result = {
            'portfolio_id': portfolio_id,
            'confidence_level': confidence_level,
            **portfolio_stats,
            'var': var,
            'cvar': cvar,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'cvar_sharpe_ratio': cvar_sharpe,
            'cvar_sortino_ratio': cvar_sortino,
            'cvar_calmar_ratio': cvar_calmar,
            'return_over_cvar': return_over_cvar_val,
            'return_over_var': return_over_var_val,
            'max_drawdown': max_dd,
            'calmar_ratio': calmar,
            **structure_metrics,
            **risk_metrics,
            **distribution_metrics,
            **tail_risk_metrics,
            **cvar_sensitivity,
            **var_sensitivity,
            'runtime_per_optimization_ms': runtime_total * 1000,
            'scenario_construction_time_ms': scenario_time,
            'solver_time_ms': solver_time,
            'optimization_status': opt_info.get('status', 'unknown')
        }
        
        return result, runtime_total
        
    except Exception as e:
        warnings.warn(f"Error optimizing portfolio {portfolio_id}: {e}")
        return {}, 0.0


def run_cvar_optimization(
    config_path: Optional[Union[str, Path]] = None,
    config_dict: Optional[Dict] = None,
    max_portfolios: Optional[int] = 100
) -> Dict:
    """
    Run CVaR optimization pipeline.
    
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
    cvar_settings = config['cvar_settings']
    modules = config.get('modules', {})
    outputs = config['outputs']
    report_sections = config.get('report_sections', [])
    
    print("Loading data...")
    # Load data
    prices = load_panel_prices(inputs['panel_price_path'])
    baseline_portfolios = load_baseline_portfolios(inputs['baseline_portfolios_path'])
    risk_free_rate = inputs.get('risk_free_rate', 0.0)
    
    print("Computing returns...")
    # Compute returns
    daily_returns = compute_daily_returns(prices, method=inputs.get('return_type', 'log'))
    
    # Get data period
    data_period = f"{prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}"
    
    # Get portfolio IDs
    if isinstance(baseline_portfolios, pd.DataFrame):
        portfolio_ids = baseline_portfolios.index.tolist()
    else:
        portfolio_ids = [0]
    
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
    cvar_frontiers = []
    
    for portfolio_id in portfolio_ids:
        result, runtime = _process_single_optimization(
            daily_returns,
            baseline_portfolios,
            portfolio_id,
            cvar_settings,
            risk_free_rate,
            random_seed=cvar_settings.get('random_seed')
        )
        
        if result:
            all_results.append(result)
            all_runtimes.append(runtime)
            # Store weights (would need to be retrieved from optimization)
    
    if len(all_results) == 0:
        raise RuntimeError("No portfolios were successfully optimized")
    
    print("Computing summary statistics...")
    # Compute summary statistics
    summary_stats = _compute_summary_statistics(all_results, all_runtimes)
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame(all_results)
    
    # Generate CVaR-return frontier if requested
    if modules.get('generate_cvar_return_frontier', False):
        print("Generating CVaR-return frontier...")
        try:
            # Use first portfolio's assets for frontier
            assets = daily_returns.columns.tolist()
            returns_subset = daily_returns[assets]
            
            scenario_settings = cvar_settings.get('scenario_generation', {})
            estimation_windows = scenario_settings.get('estimation_windows', [252])
            scenario_matrix, _ = generate_scenario_matrix(
                returns_subset,
                source=scenario_settings.get('source', 'historical'),
                estimation_window=estimation_windows[0],
                random_seed=cvar_settings.get('random_seed')
            )
            
            expected_returns = pd.Series(
                scenario_matrix.mean(axis=0),
                index=returns_subset.columns
            )
            
            confidence_levels = cvar_settings.get('confidence_levels', [0.95])
            target_return_grid = cvar_settings.get('risk_return_tradeoff', {}).get('target_return_grid', [])
            
            frontier = generate_cvar_return_frontier(
                scenario_matrix,
                expected_returns.values,
                confidence_level=confidence_levels[0],
                target_return_grid=target_return_grid if target_return_grid else None,
                constraints=cvar_settings.get('constraints', {}),
                solver=cvar_settings.get('optimization_solver', {}).get('lp_backend', 'highs')
            )
            cvar_frontiers.append(frontier)
        except Exception as e:
            warnings.warn(f"Failed to generate CVaR-return frontier: {e}")
    
    print("Saving results...")
    # Save outputs
    output_base = Path(outputs['metrics_table']).parent
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Save metrics table
    metrics_df.to_parquet(outputs['metrics_table'])
    
    # Save optimal portfolios (weights)
    if 'optimal_portfolios' in outputs:
        # Create weights DataFrame placeholder
        weights_df = pd.DataFrame(index=metrics_df['portfolio_id'])
        weights_df.to_parquet(outputs['optimal_portfolios'])
    
    # Save CVaR-return frontier
    if 'cvar_return_frontier' in outputs and len(cvar_frontiers) > 0:
        cvar_frontiers[0].to_parquet(outputs['cvar_return_frontier'])
    
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
        cvar_settings
    )
    
    print("Generating report...")
    # Generate report
    generate_report(
        metrics_df,
        outputs['summary_report'],
        cvar_settings,
        report_sections
    )
    
    print("Optimization complete!")
    
    return {
        'metrics_df': metrics_df,
        'summary_stats': summary_stats,
        'cvar_frontiers': cvar_frontiers,
        'num_portfolios': num_portfolios,
        'num_portfolios_total': num_portfolios_total
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run CVaR Portfolio Optimization'
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
    results = run_cvar_optimization(
        config_path=args.config,
        max_portfolios=max_portfolios
    )
    print(f"Optimized {results['num_portfolios']:,} portfolios (out of {results['num_portfolios_total']:,} total)")

