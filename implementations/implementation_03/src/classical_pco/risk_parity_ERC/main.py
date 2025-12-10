"""
Main evaluation script for Risk Parity / Equal Risk Contribution (ERC) Portfolio Optimization.

Orchestrates the entire optimization pipeline including:
- Data loading (prices, baseline portfolios)
- Returns computation
- Covariance matrix estimation
- Risk Parity ERC optimization
- Risk contribution analysis
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
from .risk_parity_erc_optimizer import (
    compute_covariance_matrix,
    calculate_risk_contributions,
    optimize_risk_parity_portfolio
)
from .metrics import (
    compute_portfolio_statistics,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_max_drawdown,
    compute_calmar_ratio,
    compute_risk_parity_specific_metrics,
    compute_structure_metrics,
    compute_risk_metrics,
    compute_distribution_metrics,
    compute_comparison_metrics,
    compute_runtime_metrics
)
from .report_generator import generate_report


def _save_restructured_json(
    json_path: Path,
    portfolio_results: List[Dict],
    summary_stats: Dict,
    data_period: str,
    num_portfolios: int,
    rp_settings: Dict
):
    """
    Save results in the restructured JSON format.
    
    Args:
        json_path: Path to save JSON file
        portfolio_results: List of restructured portfolio results
        summary_stats: Summary statistics dictionary
        data_period: Data period string
        num_portfolios: Number of portfolios optimized
        rp_settings: Risk Parity optimization settings
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
            'task': 'risk_parity_erc_optimization',
            'data_period': data_period,
            'portfolios_optimized': num_portfolios,
            'risk_parity_settings': rp_settings,
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
    
    if 'volatility' in results_df.columns:
        portfolio_insights['avg_volatility'] = float(results_df['volatility'].mean())
        portfolio_insights['min_volatility'] = float(results_df['volatility'].min())
    
    if 'sharpe_ratio' in results_df.columns:
        portfolio_insights['avg_sharpe_ratio'] = float(results_df['sharpe_ratio'].mean())
        portfolio_insights['max_sharpe_ratio'] = float(results_df['sharpe_ratio'].max())
    
    # Risk Parity specific insights
    rp_insights = {}
    if 'risk_parity_deviation_score' in results_df.columns:
        rp_insights['avg_risk_parity_deviation'] = float(results_df['risk_parity_deviation_score'].mean())
    if 'risk_contribution_coefficient_of_variation' in results_df.columns:
        rp_insights['avg_risk_contribution_cv'] = float(results_df['risk_contribution_coefficient_of_variation'].mean())
    if 'equal_risk_gap' in results_df.columns:
        rp_insights['avg_equal_risk_gap'] = float(results_df['equal_risk_gap'].mean())
    
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
    
    # Runtime stats
    runtime_stats = {}
    if len(runtimes) > 0:
        runtime_array = np.array(runtimes) * 1000  # Convert to ms
        runtime_stats['mean_runtime_ms'] = float(np.mean(runtime_array))
        runtime_stats['p95_runtime_ms'] = float(np.percentile(runtime_array, 95))
        runtime_stats['median_runtime_ms'] = float(np.median(runtime_array))
    
    return {
        'portfolio_level_insights': portfolio_insights,
        'risk_parity_insights': rp_insights,
        'distribution_effects': distribution_effects,
        'structure_effects': structure_effects,
        'runtime_stats': runtime_stats
    }


def _process_single_optimization(
    returns: pd.DataFrame,
    baseline_portfolios: pd.DataFrame,
    portfolio_id: Union[int, str],
    rp_settings: Dict,
    risk_free_rate: float = 0.0,
    random_seed: Optional[int] = None
) -> Tuple[Dict, float]:
    """
    Process a single Risk Parity ERC optimization for a portfolio.
    
    Args:
        returns: DataFrame of daily returns
        baseline_portfolios: DataFrame with baseline portfolio information
        portfolio_id: Portfolio identifier
        rp_settings: Risk Parity optimization settings
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
        cov_settings = rp_settings.get('covariance_estimation', {})
        constraints = rp_settings.get('constraints', {})
        optimization_settings = rp_settings.get('optimization', {})
        
        # Compute covariance matrix
        cov_start = time.time()
        cov_matrix, cov_time = compute_covariance_matrix(
            returns_subset,
            method=cov_settings.get('method', 'sample'),
            window=cov_settings.get('estimation_windows', [252])[0] if cov_settings.get('estimation_windows') else None,
            use_shrinkage=cov_settings.get('shrinkage', {}).get('use_shrinkage', False),
            shrinkage_method=cov_settings.get('shrinkage', {}).get('method', 'ledoit_wolf')
        )
        cov_time = cov_time  # Already in ms
        
        # Get baseline portfolio weights for comparison
        baseline_weights = None
        baseline_cov = None
        if portfolio_id in baseline_portfolios.index:
            portfolio_row = baseline_portfolios.loc[portfolio_id]
            if isinstance(portfolio_row, pd.Series):
                baseline_weights = portfolio_row[assets].fillna(0.0)
                baseline_weights = baseline_weights / baseline_weights.sum() if baseline_weights.sum() > 0 else baseline_weights
                baseline_cov = cov_matrix  # Use same covariance for fair comparison
        
        # Optimize Risk Parity portfolio
        optimal_weights, opt_info, solver_time = optimize_risk_parity_portfolio(
            cov_matrix,
            constraints,
            optimization_settings
        )
        
        # Calculate risk contributions
        rc_start = time.time()
        risk_contrib, marginal_contrib, portfolio_vol = calculate_risk_contributions(
            optimal_weights.values,
            cov_matrix
        )
        rc_time = (time.time() - rc_start) * 1000
        
        # Compute portfolio returns
        portfolio_returns = (returns_subset * optimal_weights).sum(axis=1)
        
        # Compute expected returns (historical mean)
        expected_returns = returns_subset.mean() * 252  # Annualized
        
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
        
        # Risk Parity specific metrics
        rp_metrics = compute_risk_parity_specific_metrics(
            optimal_weights,
            cov_matrix
        )
        
        structure_metrics = compute_structure_metrics(
            optimal_weights,
            cov_matrix,
            returns_subset,
            baseline_weights
        )
        
        risk_metrics = compute_risk_metrics(portfolio_returns)
        
        distribution_metrics = compute_distribution_metrics(portfolio_returns)
        
        # Comparison metrics
        baseline_portfolio_returns = None
        if baseline_weights is not None:
            baseline_portfolio_returns = (returns_subset * baseline_weights).sum(axis=1)
        
        comparison_metrics = {}
        if baseline_portfolio_returns is not None and baseline_cov is not None:
            comparison_metrics = compute_comparison_metrics(
                portfolio_returns,
                baseline_portfolio_returns,
                optimal_weights,
                baseline_weights,
                cov_matrix,
                baseline_cov,
                risk_free_rate,
                returns_dataframe=returns_subset
            )
        else:
            # Still compute ERC vs equal weight
            n = len(optimal_weights)
            equal_weights = pd.Series(np.ones(n) / n, index=optimal_weights.index)
            equal_weight_returns = (returns_subset * equal_weights).sum(axis=1)
            
            eq_vol = equal_weight_returns.std() * np.sqrt(252)
            erc_vol = portfolio_returns.std() * np.sqrt(252)
            eq_sharpe = compute_sharpe_ratio(equal_weight_returns, risk_free_rate)
            erc_sharpe = sharpe
            
            comparison_metrics = {
                'erc_vs_equal_weight_volatility': erc_vol - eq_vol if not np.isnan(erc_vol) and not np.isnan(eq_vol) else np.nan,
                'erc_vs_equal_weight_sharpe': erc_sharpe - eq_sharpe if not np.isnan(erc_sharpe) and not np.isnan(eq_sharpe) else np.nan,
                'baseline_portfolio_volatility': np.nan,
                'baseline_portfolio_sharpe': np.nan,
                'baseline_portfolio_expected_return': np.nan,
                'volatility_reduction_vs_baseline': np.nan,
                'sharpe_improvement_vs_baseline': np.nan,
                'risk_contribution_improvement_vs_baseline': np.nan,
                'erc_vs_equal_weight_risk_contributions': np.nan,
                'difference_in_risk_contributions_vs_baseline': np.nan
            }
        
        runtime_total = time.time() - start_time
        
        # Combine all results
        result = {
            'portfolio_id': portfolio_id,
            'method': rp_settings.get('method', 'equal_risk_contribution'),
            **portfolio_stats,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'calmar_ratio': calmar,
            **rp_metrics,
            **structure_metrics,
            **risk_metrics,
            **distribution_metrics,
            **comparison_metrics,
            'runtime_per_optimization_ms': runtime_total * 1000,
            'covariance_estimation_time_ms': cov_time,
            'risk_contribution_calculation_time_ms': rc_time,
            'solver_time_ms': solver_time,
            'optimization_status': opt_info.get('status', 'unknown')
        }
        
        return result, runtime_total
        
    except Exception as e:
        warnings.warn(f"Error optimizing portfolio {portfolio_id}: {e}")
        import traceback
        traceback.print_exc()
        return {}, 0.0


def run_risk_parity_erc_optimization(
    config_path: Optional[Union[str, Path]] = None,
    config_dict: Optional[Dict] = None,
    max_portfolios: Optional[int] = 100
) -> Dict:
    """
    Run Risk Parity ERC optimization pipeline.
    
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
    rp_settings = config['risk_parity_settings']
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
    all_risk_contributions = []
    
    for portfolio_id in portfolio_ids:
        result, runtime = _process_single_optimization(
            daily_returns,
            baseline_portfolios,
            portfolio_id,
            rp_settings,
            risk_free_rate,
            random_seed=rp_settings.get('random_seed')
        )
        
        if result:
            all_results.append(result)
            all_runtimes.append(runtime)
            
            # Store weights and risk contributions for this portfolio
            # (We'll need to recompute them or store them in the result)
    
    if len(all_results) == 0:
        raise RuntimeError("No portfolios were successfully optimized")
    
    print("Computing summary statistics...")
    # Compute summary statistics
    summary_stats = _compute_summary_statistics(all_results, all_runtimes)
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame(all_results)
    
    # Extract weights and risk contributions
    # We need to recompute them or store them during optimization
    print("Extracting optimal portfolios and risk contributions...")
    weights_list = []
    risk_contrib_list = []
    
    for portfolio_id in portfolio_ids[:len(all_results)]:
        try:
            # Get assets for this portfolio
            if portfolio_id in baseline_portfolios.index:
                portfolio_assets = baseline_portfolios.loc[portfolio_id]
                if isinstance(portfolio_assets, pd.Series):
                    assets = portfolio_assets.index.tolist()
                else:
                    assets = portfolio_assets
            else:
                assets = daily_returns.columns.tolist()
            
            assets = [a for a in assets if a in daily_returns.columns]
            if len(assets) == 0:
                continue
            
            returns_subset = daily_returns[assets]
            
            # Recompute covariance and optimize to get weights
            cov_settings = rp_settings.get('covariance_estimation', {})
            cov_matrix, _ = compute_covariance_matrix(
                returns_subset,
                method=cov_settings.get('method', 'sample'),
                window=cov_settings.get('estimation_windows', [252])[0] if cov_settings.get('estimation_windows') else None,
                use_shrinkage=cov_settings.get('shrinkage', {}).get('use_shrinkage', False),
                shrinkage_method=cov_settings.get('shrinkage', {}).get('method', 'ledoit_wolf')
            )
            
            constraints = rp_settings.get('constraints', {})
            optimization_settings = rp_settings.get('optimization', {})
            
            optimal_weights, _, _ = optimize_risk_parity_portfolio(
                cov_matrix,
                constraints,
                optimization_settings
            )
            
            risk_contrib, _, _ = calculate_risk_contributions(
                optimal_weights.values,
                cov_matrix
            )
            
            weights_list.append(optimal_weights)
            risk_contrib_list.append(pd.Series(risk_contrib, index=optimal_weights.index))
            
        except Exception as e:
            warnings.warn(f"Error extracting weights for portfolio {portfolio_id}: {e}")
    
    print("Saving results...")
    # Save outputs
    output_base = Path(outputs['metrics_table']).parent
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Save metrics table
    metrics_df.to_parquet(outputs['metrics_table'])
    
    # Save optimal portfolios (weights)
    if 'optimal_portfolios' in outputs and len(weights_list) > 0:
        weights_df = pd.DataFrame(weights_list, index=metrics_df['portfolio_id'][:len(weights_list)])
        weights_df.to_parquet(outputs['optimal_portfolios'])
    
    # Save risk contributions
    if 'risk_contribution_table' in outputs and len(risk_contrib_list) > 0:
        rc_df = pd.DataFrame(risk_contrib_list, index=metrics_df['portfolio_id'][:len(risk_contrib_list)])
        rc_df.to_parquet(outputs['risk_contribution_table'])
    
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
        rp_settings
    )
    
    print("Generating report...")
    # Generate report
    generate_report(
        metrics_df,
        outputs['summary_report'],
        rp_settings,
        report_sections
    )
    
    print("Optimization complete!")
    
    return {
        'metrics_df': metrics_df,
        'summary_stats': summary_stats,
        'num_portfolios': num_portfolios,
        'num_portfolios_total': num_portfolios_total
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run Risk Parity ERC Portfolio Optimization'
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
    results = run_risk_parity_erc_optimization(
        config_path=args.config,
        max_portfolios=max_portfolios
    )
    print(f"Optimized {results['num_portfolios']:,} portfolios (out of {results['num_portfolios_total']:,} total)")

