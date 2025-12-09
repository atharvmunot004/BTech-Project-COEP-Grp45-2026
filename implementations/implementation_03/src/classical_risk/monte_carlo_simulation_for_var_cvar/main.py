"""
Main evaluation script for Monte Carlo Simulation VaR and CVaR.

Orchestrates the entire VaR/CVaR evaluation pipeline including:
- Data loading
- Returns computation
- Rolling VaR and CVaR calculation via Monte Carlo simulation
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

# Suppress RuntimeWarning about module import in multiprocessing workers
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*found in sys.modules.*')

from .returns import (
    load_panel_prices,
    load_portfolio_weights,
    compute_daily_returns,
    compute_portfolio_returns
)
from .monte_carlo_calculator import (
    compute_rolling_var,
    compute_rolling_cvar,
    align_returns_and_var,
    align_returns_and_cvar
)
from .backtesting import compute_accuracy_metrics, detect_cvar_violations
from .metrics import (
    compute_tail_metrics,
    compute_cvar_tail_metrics,
    compute_structure_metrics,
    compute_distribution_metrics,
    compute_runtime_metrics
)
from .report_generator import generate_report
from .time_sliced_metrics import compute_time_sliced_metrics


def _restructure_results_by_portfolio(
    all_results: List[Dict],
    aligned_data_dict: Dict[int, Dict],
    prices: pd.DataFrame
) -> List[Dict]:
    """
    Restructure results grouped by portfolio.
    
    Args:
        all_results: List of flat result dictionaries
        aligned_data_dict: Dictionary mapping portfolio_id to aligned data
        prices: Original price DataFrame for date range
        
    Returns:
        List of portfolio result dictionaries with nested structure
    """
    # Group results by portfolio_id
    portfolios_dict = {}
    
    for result in all_results:
        portfolio_id = result['portfolio_id']
        
        if portfolio_id not in portfolios_dict:
            portfolios_dict[portfolio_id] = {
                'portfolio_id': portfolio_id,
                'structure': {
                    'portfolio_size': result.get('portfolio_size', result.get('num_active_assets', 0)),
                    'num_active_assets': result.get('num_active_assets', result.get('portfolio_size', 0)),
                    'hhi': result.get('hhi_concentration', np.nan),
                    'effective_assets': result.get('effective_number_of_assets', np.nan),
                    'covariance_condition_number': result.get('covariance_condition_number', np.nan)
                },
                'distribution': {
                    'skewness': result.get('skewness', np.nan),
                    'kurtosis': result.get('kurtosis', np.nan),
                    'jarque_bera_p_value': result.get('jarque_bera_p_value', np.nan),
                    'jarque_bera_statistic': result.get('jarque_bera_statistic', np.nan)
                },
                'var_evaluations': []
            }
        
        # Create VaR/CVaR evaluation entry
        var_eval = {
            'confidence_level': result['confidence_level'],
            'horizon': result['horizon'],
            'estimation_window': result['estimation_window'],
            'global_metrics': {
                'hit_rate': result.get('hit_rate', np.nan),
                'num_violations': result.get('num_violations', 0),
                'expected_violations': result.get('expected_violations', np.nan),
                'violation_ratio': result.get('violation_ratio', np.nan),
                'accuracy_tests': {
                    'kupiec_p_value': result.get('kupiec_unconditional_coverage', np.nan),
                    'kupiec_statistic': result.get('kupiec_test_statistic', np.nan),
                    'kupiec_reject_null': result.get('kupiec_reject_null', False),
                    'christoffersen_independence_p': result.get('christoffersen_independence', np.nan),
                    'christoffersen_independence_statistic': result.get('christoffersen_independence_statistic', np.nan),
                    'christoffersen_independence_reject': result.get('christoffersen_independence_reject_null', False),
                    'christoffersen_cc_p': result.get('christoffersen_conditional_coverage', np.nan),
                    'christoffersen_cc_statistic': result.get('christoffersen_conditional_coverage_statistic', np.nan),
                    'christoffersen_cc_reject': result.get('christoffersen_conditional_coverage_reject_null', False),
                    'traffic_light_zone': result.get('traffic_light_zone', 'unknown')
                },
                'tail_metrics': {
                    'mean_exceedance': result.get('mean_exceedance', np.nan),
                    'max_exceedance': result.get('max_exceedance', np.nan),
                    'std_exceedance': result.get('std_exceedance', np.nan),
                    'quantile_loss_score': result.get('quantile_loss_score', np.nan),
                    'rmse': result.get('rmse_var_vs_losses', np.nan),
                    'cvar_mean_exceedance': result.get('cvar_mean_exceedance', np.nan),
                    'cvar_max_exceedance': result.get('cvar_max_exceedance', np.nan),
                    'cvar_std_exceedance': result.get('cvar_std_exceedance', np.nan)
                },
                'runtime': {
                    'runtime_ms': result.get('var_runtime_ms', np.nan),
                    'simulation_time_ms': result.get('simulation_time_ms', np.nan),
                    'p95_runtime_ms': result.get('p95_runtime_ms', np.nan),
                    'median_runtime_ms': result.get('median_runtime_ms', np.nan)
                }
            },
            'time_sliced_metrics': []
        }
        
        # Add time-sliced metrics if available
        if portfolio_id in aligned_data_dict:
            aligned_data = aligned_data_dict[portfolio_id]
            key = f"{result['confidence_level']}_{result['horizon']}_{result['estimation_window']}"
            if key in aligned_data:
                time_slices = compute_time_sliced_metrics(
                    aligned_data[key]['returns'],
                    aligned_data[key]['var'],
                    cvar_series=aligned_data[key].get('cvar'),
                    confidence_level=result['confidence_level'],
                    slice_by='year'
                )
                var_eval['time_sliced_metrics'] = time_slices
        
        portfolios_dict[portfolio_id]['var_evaluations'].append(var_eval)
    
    return list(portfolios_dict.values())


def _save_restructured_json(
    json_path: Path,
    portfolio_results: List[Dict],
    summary_stats: Dict,
    data_period: str,
    num_portfolios: int,
    confidence_levels: List[float],
    horizons: List[int],
    estimation_windows: List[int],
    monte_carlo_settings: Optional[Dict] = None
):
    """
    Save results in the restructured JSON format.
    
    Args:
        json_path: Path to save JSON file
        portfolio_results: List of restructured portfolio results
        summary_stats: Summary statistics dictionary
        data_period: Data period string
        num_portfolios: Number of portfolios evaluated
        confidence_levels: List of confidence levels
        horizons: List of horizons
        estimation_windows: List of estimation windows
        monte_carlo_settings: Monte Carlo simulation settings
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
            'task': 'monte_carlo_var_cvar_backtesting',
            'data_period': data_period,
            'portfolios_evaluated': num_portfolios,
            'confidence_levels': confidence_levels,
            'horizons': horizons,
            'estimation_windows': estimation_windows,
            'monte_carlo_settings': monte_carlo_settings or {},
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
    
    # Average violation ratios by confidence level
    for cl in [0.95, 0.99]:
        cl_results = results_df[results_df['confidence_level'] == cl]
        if len(cl_results) > 0:
            portfolio_insights[f'avg_violation_ratio_{int(cl*100)}'] = float(
                cl_results['violation_ratio'].mean()
            )
    
    # Traffic light zones
    if 'traffic_light_zone' in results_df.columns:
        zone_counts = results_df['traffic_light_zone'].value_counts()
        total = len(results_df)
        portfolio_insights['percent_red_zone'] = float(zone_counts.get('red', 0) / total) if total > 0 else 0.0
        portfolio_insights['percent_yellow_zone'] = float(zone_counts.get('yellow', 0) / total) if total > 0 else 0.0
        portfolio_insights['percent_green_zone'] = float(zone_counts.get('green', 0) / total) if total > 0 else 0.0
    
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
    if 'hhi_concentration' in results_df.columns and 'violation_ratio' in results_df.columns:
        corr = results_df['hhi_concentration'].corr(results_df['violation_ratio'])
        if not np.isnan(corr):
            structure_effects['correlation_hhi_vs_violation_ratio'] = float(corr)
    
    if 'effective_number_of_assets' in results_df.columns and 'rmse_var_vs_losses' in results_df.columns:
        corr = results_df['effective_number_of_assets'].corr(results_df['rmse_var_vs_losses'])
        if not np.isnan(corr):
            structure_effects['correlation_enc_vs_rmse'] = float(corr)
    
    # Runtime stats
    runtime_stats = {}
    if len(runtimes) > 0:
        runtime_array = np.array(runtimes) * 1000  # Convert to ms
        runtime_stats['mean_runtime_ms'] = float(np.mean(runtime_array))
        runtime_stats['p95_runtime_ms'] = float(np.percentile(runtime_array, 95))
        runtime_stats['median_runtime_ms'] = float(np.median(runtime_array))
    
    # CVaR metrics summary
    cvar_insights = {}
    if 'cvar_mean_exceedance' in results_df.columns:
        cvar_insights['avg_cvar_mean_exceedance'] = float(results_df['cvar_mean_exceedance'].mean())
    if 'cvar_max_exceedance' in results_df.columns:
        cvar_insights['avg_cvar_max_exceedance'] = float(results_df['cvar_max_exceedance'].mean())
    
    return {
        'portfolio_level_insights': portfolio_insights,
        'distribution_effects': distribution_effects,
        'structure_effects': structure_effects,
        'runtime_stats': runtime_stats,
        'cvar_insights': cvar_insights
    }


def _process_single_portfolio(
    portfolio_data: Tuple[int, Tuple, pd.Series],
    daily_returns: pd.DataFrame,
    confidence_levels: List[float],
    horizons: List[int],
    estimation_windows: List[int],
    num_simulations: int = 10000,
    distribution_type: str = 'multivariate_normal',
    random_seed: Optional[int] = None,
    compute_time_slices: bool = True
) -> Tuple[List[Dict], float, Dict]:
    """
    Process a single portfolio - worker function for multiprocessing.
    
    Note: random_seed is applied per-portfolio, but each worker process
    may have different random states due to multiprocessing.
    """
    """
    Process a single portfolio and return all results for all configurations.
    
    This is a worker function designed to be called in parallel.
    
    Args:
        portfolio_data: Tuple of (portfolio_idx, portfolio_id, portfolio_weights)
        daily_returns: DataFrame of daily returns
        confidence_levels: List of confidence levels
        horizons: List of horizons
        estimation_windows: List of estimation windows
        num_simulations: Number of Monte Carlo simulations
        distribution_type: Distribution assumption
        random_seed: Random seed for reproducibility
        compute_time_slices: Whether to compute time-sliced metrics
        
    Returns:
        Tuple of (list of result dictionaries, simulation runtime in seconds, 
                 dict with aligned returns/VaR/CVaR for time slicing)
    """
    portfolio_idx, portfolio_id, portfolio_weights = portfolio_data
    
    results = []
    simulation_runtimes = []  # Track simulation time per configuration
    aligned_data = {}  # Store aligned returns/VaR/CVaR for time slicing
    
    try:
        # Compute portfolio returns
        portfolio_returns = compute_portfolio_returns(
            daily_returns,
            portfolio_weights,
            align_assets=True
        )
    except Exception as e:
        return results, 0.0, {}
    
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
                    # Compute rolling VaR and CVaR - track time
                    sim_start_time = time.time()
                    rolling_var = compute_rolling_var(
                        daily_returns,
                        portfolio_weights,
                        window=window,
                        confidence_level=confidence_level,
                        horizon=horizon,
                        num_simulations=num_simulations,
                        distribution_type=distribution_type,
                        random_seed=random_seed
                    )
                    rolling_cvar = compute_rolling_cvar(
                        daily_returns,
                        portfolio_weights,
                        window=window,
                        confidence_level=confidence_level,
                        horizon=horizon,
                        num_simulations=num_simulations,
                        distribution_type=distribution_type,
                        random_seed=random_seed
                    )
                    sim_runtime = time.time() - sim_start_time
                    simulation_runtimes.append(sim_runtime)
                    
                    # Align returns and VaR/CVaR
                    aligned_returns, aligned_var = align_returns_and_var(
                        portfolio_returns,
                        rolling_var
                    )
                    _, aligned_cvar = align_returns_and_cvar(
                        portfolio_returns,
                        rolling_cvar
                    )
                    
                    if len(aligned_returns) == 0:
                        continue
                    
                    # Store aligned data for time slicing
                    if compute_time_slices:
                        key = f"{confidence_level}_{horizon}_{window}"
                        aligned_data[key] = {
                            'returns': aligned_returns,
                            'var': aligned_var,
                            'cvar': aligned_cvar,
                            'confidence_level': confidence_level
                        }
                    
                    # Compute accuracy metrics
                    accuracy_metrics = compute_accuracy_metrics(
                        aligned_returns,
                        aligned_var,
                        confidence_level=confidence_level
                    )
                    
                    # Compute tail metrics for VaR
                    tail_metrics = compute_tail_metrics(
                        aligned_returns,
                        aligned_var,
                        confidence_level=confidence_level
                    )
                    
                    # Compute CVaR tail metrics
                    cvar_tail_metrics = compute_cvar_tail_metrics(
                        aligned_returns,
                        aligned_cvar,
                        aligned_var,
                        confidence_level=confidence_level
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
                        'var_runtime_ms': sim_runtime * 1000,
                        'simulation_time_ms': sim_runtime * 1000,
                        **accuracy_metrics,
                        **tail_metrics,
                        **cvar_tail_metrics,
                        **structure_metrics,
                        **distribution_metrics
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    continue
    
    # Return total runtime per portfolio
    sim_runtime_total = sum(simulation_runtimes) if len(simulation_runtimes) > 0 else 0.0
    
    return results, sim_runtime_total, aligned_data


def evaluate_monte_carlo_var_cvar(
    config_path: Optional[Union[str, Path]] = None,
    config_dict: Optional[Dict] = None,
    n_jobs: Optional[int] = None,
    max_portfolios: Optional[int] = 100
) -> pd.DataFrame:
    """
    Main function to evaluate VaR and CVaR using Monte Carlo simulation for multiple portfolios.
    
    Args:
        config_path: Path to JSON configuration file
        config_dict: Configuration dictionary (if not loading from file)
        n_jobs: Number of parallel workers (default: number of CPU cores)
        max_portfolios: Maximum number of portfolios to process (default: 100, None for all)
        
    Returns:
        DataFrame with all computed metrics
    """
    # Load configuration
    if config_dict is None:
        if config_path is None:
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
    print("MONTE CARLO SIMULATION FOR VAR/CVAR EVALUATION")
    print("=" * 80)
    print(f"\nLoading data...")
    print(f"  Panel prices: {panel_price_path}")
    print(f"  Portfolio weights: {portfolio_weights_path}")
    
    # Load data
    prices = load_panel_prices(panel_price_path)
    portfolio_weights_df = load_portfolio_weights(portfolio_weights_path)
    
    # Get data period
    data_period = f"{prices.index.min().strftime('%Y-%m-%d')} to {prices.index.max().strftime('%Y-%m-%d')}"
    
    print(f"\nLoaded:")
    print(f"  Prices: {len(prices)} dates, {len(prices.columns)} assets")
    print(f"  Portfolios: {len(portfolio_weights_df)} portfolios")
    print(f"  Data period: {data_period}")
    
    # Compute daily returns
    print(f"\nComputing daily returns...")
    daily_returns = compute_daily_returns(prices, method='log')
    print(f"  Daily returns: {len(daily_returns)} dates")
    
    # Get Monte Carlo settings
    mc_settings = config['monte_carlo_settings']
    confidence_levels = mc_settings['confidence_levels']
    horizons = mc_settings['horizons']
    estimation_windows = mc_settings['estimation_windows']
    num_simulations = mc_settings.get('num_simulations', 10000)
    distribution_type = mc_settings.get('distribution_type', 'multivariate_normal')
    random_seed = mc_settings.get('random_seed', None)
    
    print(f"\nMonte Carlo Settings:")
    print(f"  Number of simulations: {num_simulations:,}")
    print(f"  Distribution type: {distribution_type}")
    print(f"  Random seed: {random_seed}")
    
    # Initialize results list
    all_results = []
    runtimes = []
    aligned_data_dict = {}
    
    # Process each portfolio
    print(f"\nEvaluating {len(portfolio_weights_df)} portfolios...")
    print(f"  Confidence levels: {confidence_levels}")
    print(f"  Horizons: {horizons} days")
    print(f"  Estimation windows: {estimation_windows} days")
    
    num_portfolios_total = len(portfolio_weights_df)
    
    # Limit number of portfolios if specified
    if max_portfolios is not None and max_portfolios > 0:
        num_portfolios = min(num_portfolios_total, max_portfolios)
        portfolio_weights_df = portfolio_weights_df.iloc[:num_portfolios]
        print(f"  Limiting to first {num_portfolios:,} portfolios (out of {num_portfolios_total:,} total)")
    else:
        num_portfolios = num_portfolios_total
        print(f"  Processing all {num_portfolios:,} portfolios")
    
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
        estimation_windows=estimation_windows,
        num_simulations=num_simulations,
        distribution_type=distribution_type,
        random_seed=random_seed,
        compute_time_slices=True
    )
    
    # Process portfolios in parallel
    start_time_total = time.time()
    
    print(f"  Starting processing at {datetime.now().strftime('%H:%M:%S')}...")
    print(f"  Estimated time: ~{num_portfolios * 4 * 10 / 60:.1f} minutes (rough estimate)")
    
    if n_jobs == 1:
        print("  Running in sequential mode...")
        for portfolio_idx, portfolio_data in enumerate(portfolio_data_list):
            if (portfolio_idx + 1) % 100 == 0 or (portfolio_idx + 1) in [1, 10, 50, 500, 1000, 5000, 10000]:
                print(f"  Processing portfolio {portfolio_idx + 1:,}/{num_portfolios:,} ({100*(portfolio_idx+1)/num_portfolios:.1f}%)...")
            
            results, runtime, aligned_data = worker_func(portfolio_data)
            all_results.extend(results)
            runtimes.append(runtime)
            portfolio_id = portfolio_data[1]
            if aligned_data:
                aligned_data_dict[portfolio_id] = aligned_data
    else:
        print(f"  Running in parallel mode with {n_jobs} workers...")
        print(f"  Note: Monte Carlo simulation is computationally intensive. This may take several minutes...")
        completed = 0
        
        with Pool(processes=n_jobs) as pool:
            results_iter = pool.imap(worker_func, portfolio_data_list, chunksize=max(1, num_portfolios // (n_jobs * 4)))
            
            # Add timeout and progress tracking
            last_progress_time = time.time()
            for results, runtime, aligned_data in results_iter:
                completed += 1
                all_results.extend(results)
                runtimes.append(runtime)
                if results and len(results) > 0:
                    portfolio_id = results[0]['portfolio_id']
                    if aligned_data:
                        aligned_data_dict[portfolio_id] = aligned_data
                
                # Print progress more frequently for first few portfolios
                elapsed = time.time() - start_time_total
                should_print = (
                    completed % 10 == 0 or 
                    completed in [1, 2, 3, 4, 5, 10, 20, 50] or
                    (completed % 100 == 0) or
                    (time.time() - last_progress_time > 30)  # Print at least every 30 seconds
                )
                
                if should_print:
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = (num_portfolios - completed) / rate if rate > 0 else 0
                    print(f"  Completed {completed:,}/{num_portfolios:,} ({100*completed/num_portfolios:.1f}%) | "
                          f"Rate: {rate:.2f} portfolios/sec | "
                          f"Elapsed: {elapsed/60:.1f} min | "
                          f"Remaining: {remaining/60:.1f} min")
                    last_progress_time = time.time()
    
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
    
    # Restructure results by portfolio
    portfolio_results = _restructure_results_by_portfolio(
        all_results,
        aligned_data_dict,
        prices
    )
    
    # Compute summary statistics
    summary_stats = _compute_summary_statistics(all_results, runtimes)
    
    print(f"\nCompleted evaluation of {len(results_df)} portfolio-configuration combinations")
    print(f"  Total runtime: {total_runtime/60:.2f} minutes ({total_runtime:.2f} seconds)")
    
    if len(runtimes) > 0:
        sim_runtime_total = sum(runtimes)
        avg_sim_runtime_per_portfolio = sim_runtime_total / num_portfolios if num_portfolios > 0 else 0
        print(f"  Simulation runtime: {sim_runtime_total/60:.2f} minutes ({sim_runtime_total:.2f} seconds)")
        print(f"  Average simulation runtime: {avg_sim_runtime_per_portfolio*1000:.2f} ms per portfolio")
    
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
        elif metrics_path.suffix == '.json':
            _save_restructured_json(
                metrics_path,
                portfolio_results,
                summary_stats,
                data_period,
                num_portfolios,
                confidence_levels,
                horizons,
                estimation_windows,
                mc_settings
            )
        else:
            metrics_path = metrics_path.with_suffix('.parquet')
            results_df.to_parquet(metrics_path, index=False)
        
        print(f"  Saved: {metrics_path}")
    
    # Also save JSON if specified separately
    if 'metrics_json' in outputs:
        json_path = project_root / outputs['metrics_json']
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving metrics JSON...")
        print(f"  Path: {json_path}")
        
        _save_restructured_json(
            json_path,
            portfolio_results,
            summary_stats,
            data_period,
            num_portfolios,
            confidence_levels,
            horizons,
            estimation_windows,
            mc_settings
        )
        
        print(f"  Saved: {json_path}")
    
    # Generate report
    if 'summary_report' in outputs:
        report_path = project_root / outputs['summary_report']
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating report...")
        print(f"  Path: {report_path}")
        
        generate_report(
            results_df,
            report_path,
            monte_carlo_settings=mc_settings,
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
        description='Evaluate Monte Carlo Simulation VaR/CVaR for portfolios'
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
    parser.add_argument(
        '--max-portfolios',
        type=int,
        default=100,
        help='Maximum number of portfolios to process (default: 100, use 0 to process all)'
    )
    
    args = parser.parse_args()
    
    n_jobs = args.n_jobs
    if n_jobs == -1:
        n_jobs = None
    
    max_portfolios = args.max_portfolios
    if max_portfolios == 0:
        max_portfolios = None
    
    results_df = evaluate_monte_carlo_var_cvar(
        config_path=args.config,
        n_jobs=n_jobs,
        max_portfolios=max_portfolios
    )
    
    print(f"\nResults summary:")
    print(f"  Total rows: {len(results_df)}")
    print(f"  Columns: {len(results_df.columns)}")
    print(f"\nFirst few rows:")
    print(results_df.head())


if __name__ == "__main__":
    main()

