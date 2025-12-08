"""
Comprehensive Comparative Analysis Generator
Generates all data needed for intense comparative analysis per objectives.md
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
from datetime import datetime
import time
import json
from tqdm import tqdm

from config.config_loader import get_config
from data_loading.data_loader import DataLoader
from data_cleaning.data_cleaner import DataCleaner
from feature_engineering.returns_calculator import ReturnsCalculator
from portfolio_generation.portfolio_generator import PortfolioGenerator
from classical_risk_engine.parametric_var import ParametricVaR
from classical_risk_engine.monte_carlo_var_cvar import MonteCarloVaRCVaR
from classical_risk_engine.evt_pot import EVTPOT
from quantum_risk_engine.qae_cvar import QAECVaR
from classical_portfolio_engine.markowitz_mv import MarkowitzMV
from classical_portfolio_engine.risk_parity_erc import RiskParityERC
from classical_portfolio_engine.cvar_optimization import CVaROptimization
from quantum_portfolio_engine.qmv_qubo import QMVQUBO
from quantum_portfolio_engine.qaoa_cvar_portfolio import QAOACVaRPortfolio
from evaluation_and_backtesting.backtester import RollingWindowBacktester
from evaluation_and_backtesting.metrics_calculator import MetricsCalculator


def generate_risk_comparison_data(returns_df, portfolios_df, symbols):
    """
    Objective 1: Quantum vs Classical Risk Assessment
    - Benchmark QAE vs Monte Carlo for VaR/CVaR
    - Measure error vs sample size (O(1/N) vs O(1/√N))
    - Runtime comparisons
    """
    print("\n" + "="*80)
    print("OBJECTIVE 1: Quantum vs Classical Risk Assessment")
    print("="*80)
    
    results = []
    sample_portfolios = portfolios_df.sample(n=50, random_state=42).values
    
    # Test different sample sizes for Monte Carlo
    mc_sample_sizes = [1000, 5000, 10000, 50000, 100000]
    confidence_levels = [0.95, 0.99]
    
    print("\n[1.1] Classical Monte Carlo - Sample Size Analysis...")
    for sample_size in tqdm(mc_sample_sizes, desc="MC Sample Sizes"):
        mc = MonteCarloVaRCVaR(num_mc_paths=sample_size, random_seed=42)
        
        for i, weights in enumerate(sample_portfolios[:20]):
            start_time = time.time()
            result = mc.calculate(returns_df, weights, confidence_levels=confidence_levels)
            runtime = time.time() - start_time
            
            for conf_level in confidence_levels:
                results.append({
                    'method': 'monte_carlo',
                    'sample_size': sample_size,
                    'portfolio_id': i,
                    'confidence_level': conf_level,
                    'var': result['var_mc_estimates'].get(conf_level, None),
                    'cvar': result['cvar_mc_estimates'].get(conf_level, None),
                    'runtime': runtime,
                    'num_qubits': None,
                    'circuit_depth': None
                })
    
    print("\n[1.2] Quantum QAE - Analysis...")
    qae = QAECVaR(num_shots=8192)
    
    for i, weights in enumerate(tqdm(sample_portfolios[:20], desc="QAE Portfolios")):
        portfolio_returns = np.dot(returns_df.values, weights)
        portfolio_loss_series = pd.Series(-portfolio_returns, index=returns_df.index)
        
        start_time = time.time()
        result = qae.calculate(portfolio_loss_series, confidence_levels=confidence_levels)
        runtime = time.time() - start_time
        
        for conf_level in confidence_levels:
            results.append({
                'method': 'qae',
                'sample_size': 8192,  # Shots
                'portfolio_id': i,
                'confidence_level': conf_level,
                'var': None,
                'cvar': result['cvar_qae_estimates'].get(conf_level, None),
                'runtime': runtime,
                'num_qubits': result.get('num_qubits', None),
                'circuit_depth': result.get('circuit_depth', None)
            })
    
    print("\n[1.3] Classical Parametric VaR...")
    parametric = ParametricVaR(confidence_levels=confidence_levels)
    
    for i, weights in enumerate(tqdm(sample_portfolios[:20], desc="Parametric VaR")):
        start_time = time.time()
        result = parametric.calculate(returns_df, weights)
        runtime = time.time() - start_time
        
        for conf_level in confidence_levels:
            results.append({
                'method': 'parametric_var',
                'sample_size': None,
                'portfolio_id': i,
                'confidence_level': conf_level,
                'var': result['var_estimates'].get(conf_level, None),
                'cvar': None,
                'runtime': runtime,
                'num_qubits': None,
                'circuit_depth': None
            })
    
    print("\n[1.4] EVT POT...")
    evt = EVTPOT()
    
    for i, weights in enumerate(tqdm(sample_portfolios[:20], desc="EVT POT")):
        portfolio_returns = np.dot(returns_df.values, weights)
        portfolio_loss_series = pd.Series(-portfolio_returns, index=returns_df.index)
        
        start_time = time.time()
        result = evt.calculate(portfolio_loss_series, confidence_levels=confidence_levels)
        runtime = time.time() - start_time
        
        for conf_level in confidence_levels:
            results.append({
                'method': 'evt_pot',
                'sample_size': None,
                'portfolio_id': i,
                'confidence_level': conf_level,
                'var': result['var_evt_estimates'].get(conf_level, None),
                'cvar': result['cvar_evt_estimates'].get(conf_level, None),
                'runtime': runtime,
                'num_qubits': None,
                'circuit_depth': None
            })
    
    risk_df = pd.DataFrame(results)
    return risk_df


def generate_portfolio_optimization_data(returns_df, mean_returns, cov_matrix, symbols):
    """
    Objective 2: Quantum-Enhanced Portfolio Optimization
    - Compare QMV/QAOA vs classical methods
    - Evaluate scalability
    - Compare Sharpe ratio, drawdown, solution sparsity
    """
    print("\n" + "="*80)
    print("OBJECTIVE 2: Quantum-Enhanced Portfolio Optimization")
    print("="*80)
    
    results = []
    
    # Generate scenario matrix for CVaR optimization
    num_scenarios = 10000
    scenario_returns = np.random.multivariate_normal(
        mean_returns.values,
        cov_matrix.values,
        size=num_scenarios
    )
    
    print("\n[2.1] Classical Methods...")
    
    # Markowitz
    print("  - Markowitz Mean-Variance...")
    try:
        mv = MarkowitzMV(risk_aversion=1.0)
        start_time = time.time()
        mv_result = mv.optimize(mean_returns, cov_matrix)
        runtime = time.time() - start_time
        
        results.append({
            'method': 'markowitz_mv',
            'weights': mv_result['optimal_weights_mv'],
            'expected_return': mv_result['diagnostics']['expected_return'],
            'volatility': mv_result['diagnostics']['volatility'],
            'sharpe_ratio': mv_result['diagnostics']['sharpe_ratio'],
            'runtime': runtime,
            'num_assets': len(symbols),
            'sparsity': np.sum(np.abs(mv_result['optimal_weights_mv']) > 0.01) / len(symbols)
        })
    except Exception as e:
        print(f"    Error: {e}")
    
    # Risk Parity
    print("  - Risk Parity/ERC...")
    try:
        rp = RiskParityERC()
        start_time = time.time()
        rp_result = rp.optimize(cov_matrix)
        runtime = time.time() - start_time
        
        portfolio_return = mean_returns @ rp_result['erc_weights']
        portfolio_vol = np.sqrt(rp_result['erc_weights'] @ cov_matrix.values @ rp_result['erc_weights'])
        sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
        
        results.append({
            'method': 'risk_parity_erc',
            'weights': rp_result['erc_weights'],
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe,
            'runtime': runtime,
            'num_assets': len(symbols),
            'sparsity': np.sum(np.abs(rp_result['erc_weights']) > 0.01) / len(symbols)
        })
    except Exception as e:
        print(f"    Error: {e}")
    
    # CVaR Optimization
    print("  - CVaR Optimization...")
    try:
        cvar_opt = CVaROptimization(confidence_level=0.95)
        start_time = time.time()
        cvar_result = cvar_opt.optimize(scenario_returns)
        runtime = time.time() - start_time
        
        portfolio_return = np.mean(scenario_returns @ cvar_result['optimal_weights_cvar'])
        portfolio_vol = np.std(scenario_returns @ cvar_result['optimal_weights_cvar'])
        sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
        
        results.append({
            'method': 'cvar_optimization',
            'weights': cvar_result['optimal_weights_cvar'],
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe,
            'cvar': cvar_result['cvar_value'],
            'runtime': runtime,
            'num_assets': len(symbols),
            'sparsity': np.sum(np.abs(cvar_result['optimal_weights_cvar']) > 0.01) / len(symbols)
        })
    except Exception as e:
        print(f"    Error: {e}")
    
    print("\n[2.2] Quantum Methods...")
    
    # QMV
    print("  - Quantum QMV/QUBO...")
    try:
        qmv = QMVQUBO(risk_aversion=1.0)
        start_time = time.time()
        qmv_result = qmv.optimize(mean_returns, cov_matrix)
        runtime = time.time() - start_time
        
        results.append({
            'method': 'qmv_qubo',
            'weights': qmv_result['optimal_or_near_optimal_weights_qmv'],
            'expected_return': qmv_result['risk_return_tradeoff_metrics']['expected_return'],
            'volatility': qmv_result['risk_return_tradeoff_metrics']['volatility'],
            'sharpe_ratio': qmv_result['risk_return_tradeoff_metrics']['sharpe_ratio'],
            'runtime': runtime,
            'num_assets': len(symbols),
            'num_qubits': qmv_result.get('num_qubits', None),
            'circuit_depth': qmv_result.get('circuit_depth', None),
            'sparsity': np.sum(np.abs(qmv_result['optimal_or_near_optimal_weights_qmv']) > 0.01) / len(symbols)
        })
    except Exception as e:
        print(f"    Error: {e}")
    
    # QAOA CVaR Portfolio
    print("  - QAOA CVaR Portfolio...")
    try:
        qaoa = QAOACVaRPortfolio(p_layers=2, shots=8192)
        start_time = time.time()
        qaoa_result = qaoa.optimize(scenario_returns, confidence_level=0.95)
        runtime = time.time() - start_time
        
        results.append({
            'method': 'qaoa_cvar_portfolio',
            'weights': qaoa_result['qaoa_optimal_bitstring_or_weights'],
            'expected_return': qaoa_result['risk_return_metrics']['expected_return'],
            'volatility': qaoa_result['risk_return_metrics']['volatility'],
            'sharpe_ratio': qaoa_result['risk_return_metrics'].get('sharpe_ratio', 0),
            'runtime': runtime,
            'num_assets': len(symbols),
            'num_qubits': qaoa_result.get('num_qubits', None),
            'circuit_depth': qaoa_result.get('circuit_depth', None),
            'sparsity': np.sum(np.abs(qaoa_result['qaoa_optimal_bitstring_or_weights']) > 0.01) / len(symbols)
        })
    except Exception as e:
        print(f"    Error: {e}")
    
    # Backtest all methods
    print("\n[2.3] Backtesting All Methods...")
    backtester = RollingWindowBacktester()
    windows = backtester.create_rolling_windows(returns_df)
    
    if len(windows) > 0:
        for method_result in results:
            method_name = method_result['method']
            weights = method_result['weights']
            
            def method_backtest(train_returns, test_returns):
                portfolio_returns = test_returns @ weights
                metrics = MetricsCalculator.calculate_portfolio_metrics(portfolio_returns)
                return metrics
            
            backtest_results = backtester.backtest_method(
                returns_df,
                method_backtest,
                method_name,
                windows[:10]  # Use first 10 windows
            )
            
            if len(backtest_results) > 0:
                method_result['avg_sharpe'] = backtest_results['sharpe_ratio'].mean()
                method_result['avg_return'] = backtest_results['annualized_return'].mean()
                method_result['avg_volatility'] = backtest_results['annualized_volatility'].mean()
                method_result['avg_max_drawdown'] = backtest_results['max_drawdown'].mean()
                method_result['num_windows'] = len(backtest_results)
    
    portfolio_df = pd.DataFrame(results)
    return portfolio_df


def generate_scalability_analysis(returns_df, mean_returns, cov_matrix, symbols):
    """
    Scalability analysis: N = 10, 20, 30, 40, 50 assets
    """
    print("\n" + "="*80)
    print("SCALABILITY ANALYSIS: N = 10, 20, 30, 40, 50 assets")
    print("="*80)
    
    results = []
    asset_counts = [10, 20, 30, 40, min(50, len(symbols))]
    
    for n_assets in asset_counts:
        if n_assets > len(symbols):
            continue
        
        print(f"\nTesting with {n_assets} assets...")
        subset_symbols = symbols[:n_assets]
        subset_returns = returns_df[subset_symbols]
        subset_mean = mean_returns[subset_symbols]
        subset_cov = cov_matrix.loc[subset_symbols, subset_symbols]
        
        # Markowitz
        try:
            mv = MarkowitzMV(risk_aversion=1.0)
            start_time = time.time()
            mv_result = mv.optimize(subset_mean, subset_cov)
            runtime = time.time() - start_time
            
            results.append({
                'method': 'markowitz_mv',
                'num_assets': n_assets,
                'runtime': runtime,
                'sharpe_ratio': mv_result['diagnostics']['sharpe_ratio']
            })
        except Exception as e:
            print(f"  Markowitz failed: {e}")
        
        # QMV
        try:
            qmv = QMVQUBO(risk_aversion=1.0)
            start_time = time.time()
            qmv_result = qmv.optimize(subset_mean, subset_cov)
            runtime = time.time() - start_time
            
            results.append({
                'method': 'qmv_qubo',
                'num_assets': n_assets,
                'runtime': runtime,
                'sharpe_ratio': qmv_result['risk_return_tradeoff_metrics']['sharpe_ratio']
            })
        except Exception as e:
            print(f"  QMV failed: {e}")
    
    scalability_df = pd.DataFrame(results)
    return scalability_df


def main():
    """Generate comprehensive comparative analysis data."""
    print("="*80)
    print("COMPREHENSIVE COMPARATIVE ANALYSIS GENERATOR")
    print("="*80)
    
    # Load and prepare data
    print("\n[0] Loading and preparing data...")
    data_loader = DataLoader()
    symbols = data_loader.get_available_symbols()[:10]
    data_dict = data_loader.load_multiple_stocks(symbols)
    
    data_cleaner = DataCleaner()
    panel_data = data_cleaner.clean(data_dict)
    prices_df = panel_data['adj_close']
    
    returns_calc = ReturnsCalculator()
    returns_df = returns_calc.calculate_returns(prices_df)
    mean_returns = returns_calc.calculate_mean_returns(returns_df)
    cov_matrix = returns_calc.calculate_covariance_matrix(returns_df)
    
    # Load portfolios
    portfolio_gen = PortfolioGenerator()
    portfolios_df = portfolio_gen.generate_and_save(len(symbols), symbols)
    
    # Generate all analysis data
    risk_df = generate_risk_comparison_data(returns_df, portfolios_df, symbols)
    portfolio_df = generate_portfolio_optimization_data(returns_df, mean_returns, cov_matrix, symbols)
    scalability_df = generate_scalability_analysis(returns_df, mean_returns, cov_matrix, symbols)
    
    # Save results
    results_dir = Path("results/comparative_analysis")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    risk_df.to_csv(results_dir / "risk_comparison.csv", index=False)
    portfolio_df.to_csv(results_dir / "portfolio_optimization_comparison.csv", index=False)
    scalability_df.to_csv(results_dir / "scalability_analysis.csv", index=False)
    
    # Save summary statistics
    summary = {
        'risk_comparison': {
            'total_comparisons': len(risk_df),
            'methods': risk_df['method'].unique().tolist(),
            'avg_runtime_by_method': risk_df.groupby('method')['runtime'].mean().to_dict(),
            'sample_sizes_tested': sorted(risk_df['sample_size'].dropna().unique().tolist())
        },
        'portfolio_optimization': {
            'methods_tested': portfolio_df['method'].unique().tolist(),
            'avg_runtime_by_method': portfolio_df.groupby('method')['runtime'].mean().to_dict(),
            'avg_sharpe_by_method': portfolio_df.groupby('method')['sharpe_ratio'].mean().to_dict()
        },
        'scalability': {
            'asset_counts_tested': sorted(scalability_df['num_assets'].unique().tolist()),
            'methods': scalability_df['method'].unique().tolist()
        }
    }
    
    with open(results_dir / "summary_statistics.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {results_dir}")
    print(f"  - risk_comparison.csv: {len(risk_df)} rows")
    print(f"  - portfolio_optimization_comparison.csv: {len(portfolio_df)} rows")
    print(f"  - scalability_analysis.csv: {len(scalability_df)} rows")
    print(f"  - summary_statistics.json")


if __name__ == "__main__":
    main()

