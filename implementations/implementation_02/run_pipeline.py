"""
Main execution script for the Quantum vs Classical Risk and Portfolio Engine.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
from datetime import datetime
import time

from config.config_loader import get_config
from data_loading.data_loader import DataLoader
from data_cleaning.data_cleaner import DataCleaner
from feature_engineering.returns_calculator import ReturnsCalculator
from portfolio_generation.portfolio_generator import PortfolioGenerator
from classical_risk_engine.parametric_var import ParametricVaR
from classical_risk_engine.monte_carlo_var_cvar import MonteCarloVaRCVaR
from classical_risk_engine.garch_volatility import GARCHVolatility
from classical_risk_engine.evt_pot import EVTPOT
from quantum_risk_engine.qae_cvar import QAECVaR
from quantum_risk_engine.qaoa_cvar_risk import QAOACVaRRisk
from quantum_risk_engine.qgan_scenario import QGANScenarioGeneration
from quantum_risk_engine.qpca_factor import QPCAFactorRisk
from classical_portfolio_engine.markowitz_mv import MarkowitzMV
from classical_portfolio_engine.black_litterman import BlackLitterman
from classical_portfolio_engine.risk_parity_erc import RiskParityERC
from classical_portfolio_engine.cvar_optimization import CVaROptimization
from quantum_portfolio_engine.qmv_qubo import QMVQUBO
from quantum_portfolio_engine.qaoa_cvar_portfolio import QAOACVaRPortfolio
from quantum_portfolio_engine.qae_cvar_portfolio import QAECVaRPortfolio
from evaluation_and_backtesting.backtester import RollingWindowBacktester
from evaluation_and_backtesting.metrics_calculator import MetricsCalculator
from experiment_tracking.experiment_logger import ExperimentLogger


def main():
    """Main execution pipeline."""
    print("=" * 80)
    print("Quantum vs Classical Risk and Portfolio Engine")
    print("=" * 80)
    
    # Initialize
    config = get_config()
    logger = ExperimentLogger(experiment_name="quantum_vs_classical_risk_portfolio")
    logger.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Step 1: Load data
    print("\n[1/8] Loading data...")
    data_loader = DataLoader()
    symbols = data_loader.get_available_symbols()[:10]  # Use first 10 stocks
    print(f"Loading {len(symbols)} stocks: {symbols}")
    
    data_dict = data_loader.load_multiple_stocks(symbols)
    print(f"Loaded {len(data_dict)} stocks")
    
    # Step 2: Clean data
    print("\n[2/8] Cleaning data...")
    data_cleaner = DataCleaner()
    panel_data = data_cleaner.clean(data_dict)
    prices_df = panel_data['adj_close']
    print(f"Cleaned data shape: {prices_df.shape}")
    
    # Step 3: Calculate returns
    print("\n[3/8] Calculating returns...")
    returns_calc = ReturnsCalculator()
    returns_df = returns_calc.calculate_returns(prices_df)
    mean_returns = returns_calc.calculate_mean_returns(returns_df)
    cov_matrix = returns_calc.calculate_covariance_matrix(returns_df)
    volatilities = returns_calc.calculate_volatilities(returns_df)
    print(f"Returns shape: {returns_df.shape}")
    print(f"Mean returns (annualized):\n{mean_returns}")
    
    # Step 4: Generate portfolios
    print("\n[4/8] Generating portfolios...")
    portfolio_gen = PortfolioGenerator()
    portfolios_df = portfolio_gen.generate_and_save(
        num_assets=len(symbols),
        symbols=symbols
    )
    print(f"Generated {len(portfolios_df)} portfolios")
    
    # Step 5: Risk analysis on sample portfolios
    print("\n[5/8] Running risk analysis...")
    sample_portfolios = portfolios_df.iloc[:100].values  # Use first 100 for speed
    
    # Classical risk methods
    print("  - Classical Parametric VaR...")
    parametric_var = ParametricVaR(confidence_levels=[0.95, 0.99])
    var_results = []
    for i, weights in enumerate(sample_portfolios[:10]):  # Test on 10 portfolios
        result = parametric_var.calculate(returns_df, weights)
        var_results.append(result)
    
    print("  - Classical Monte Carlo VaR/CVaR...")
    mc_var_cvar = MonteCarloVaRCVaR(num_mc_paths=10000, random_seed=42)
    mc_results = []
    for i, weights in enumerate(sample_portfolios[:10]):
        result = mc_var_cvar.calculate(returns_df, weights, confidence_levels=[0.95, 0.99])
        mc_results.append(result)
    
    # Quantum risk methods (simplified)
    print("  - Quantum QAE CVaR...")
    qae_cvar = QAECVaR(num_shots=8192)
    qae_results = []
    for i, weights in enumerate(sample_portfolios[:5]):  # Test on 5 portfolios
        portfolio_returns = np.dot(returns_df.values, weights)
        portfolio_loss_series = pd.Series(-portfolio_returns, index=returns_df.index)
        result = qae_cvar.calculate(portfolio_loss_series, confidence_levels=[0.95, 0.99])
        qae_results.append(result)
    
    # Step 6: Portfolio optimization
    print("\n[6/8] Running portfolio optimization...")
    
    # Classical methods
    print("  - Markowitz Mean-Variance...")
    markowitz = MarkowitzMV(risk_aversion=1.0)
    mv_result = markowitz.optimize(mean_returns, cov_matrix)
    print(f"    Optimal weights: {mv_result['optimal_weights_mv']}")
    print(f"    Expected return: {mv_result['diagnostics']['expected_return']:.4f}")
    print(f"    Volatility: {mv_result['diagnostics']['volatility']:.4f}")
    
    print("  - Risk Parity...")
    risk_parity = RiskParityERC()
    rp_result = risk_parity.optimize(cov_matrix)
    print(f"    Optimal weights: {rp_result['erc_weights']}")
    
    # Quantum methods
    print("  - Quantum QMV...")
    qmv = QMVQUBO(risk_aversion=1.0)
    qmv_result = qmv.optimize(mean_returns, cov_matrix)
    print(f"    Optimal weights: {qmv_result['optimal_or_near_optimal_weights_qmv']}")
    
    # Step 7: Backtesting
    print("\n[7/8] Running backtesting...")
    backtester = RollingWindowBacktester()
    windows = backtester.create_rolling_windows(returns_df)
    print(f"Created {len(windows)} rolling windows")
    
    if len(windows) == 0:
        print("Warning: No windows created. Skipping backtesting.")
        backtest_results = pd.DataFrame()
    else:
        # Simple backtest: equal weights portfolio
        def equal_weights_method(train_returns, test_returns):
            weights = np.ones(len(symbols)) / len(symbols)
            portfolio_returns = test_returns @ weights
            metrics = MetricsCalculator.calculate_portfolio_metrics(portfolio_returns)
            return metrics
        
        # Backtest multiple methods
        all_results = []
        
        # 1. Equal weights
        results_ew = backtester.backtest_method(
            returns_df,
            equal_weights_method,
            "equal_weights",
            windows
        )
        if len(results_ew) > 0:
            all_results.append(results_ew)
        
        # 2. Markowitz (if available)
        def markowitz_method(train_returns, test_returns):
            try:
                train_mean = train_returns.mean() * 252
                train_cov = train_returns.cov() * 252
                mv = MarkowitzMV(risk_aversion=1.0)
                result = mv.optimize(train_mean, train_cov)
                weights = result['optimal_weights_mv']
                portfolio_returns = test_returns @ weights
                metrics = MetricsCalculator.calculate_portfolio_metrics(portfolio_returns)
                return metrics
            except Exception as e:
                print(f"Markowitz method failed: {e}")
                return equal_weights_method(train_returns, test_returns)
        
        results_mv = backtester.backtest_method(
            returns_df,
            markowitz_method,
            "markowitz_mv",
            windows
        )
        if len(results_mv) > 0:
            all_results.append(results_mv)
        
        # Combine all results
        if all_results:
            backtest_results = pd.concat(all_results, ignore_index=True)
        else:
            backtest_results = pd.DataFrame()
        
        print(f"Backtest results shape: {backtest_results.shape}")
        if len(backtest_results) > 0:
            print(f"Methods tested: {backtest_results['method'].unique()}")
            print(f"Average Sharpe Ratio: {backtest_results['sharpe_ratio'].mean():.4f}")
            print(f"Average Annualized Return: {backtest_results['annualized_return'].mean():.4f}")
            print(f"Average Volatility: {backtest_results['annualized_volatility'].mean():.4f}")
    
    # Step 8: Log results
    print("\n[8/8] Logging results...")
    logger.log_params({
        'num_assets': len(symbols),
        'symbols': symbols,
        'num_portfolios': len(portfolios_df),
        'num_windows': len(windows)
    })
    
    # Aggregate metrics
    if len(backtest_results) > 0:
        avg_sharpe = backtest_results['sharpe_ratio'].mean()
        avg_return = backtest_results['annualized_return'].mean()
        avg_vol = backtest_results['annualized_volatility'].mean()
        
        logger.log_metrics({
            'avg_sharpe_ratio': avg_sharpe,
            'avg_annualized_return': avg_return,
            'avg_annualized_volatility': avg_vol
        })
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Save backtest results (only if not empty)
    if len(backtest_results) > 0:
        backtest_results.to_csv(results_dir / "backtest_results.csv", index=False)
        logger.log_artifact(str(results_dir / "backtest_results.csv"))
        print(f"Saved backtest results with {len(backtest_results)} rows")
    else:
        # Create empty file with headers
        empty_df = pd.DataFrame(columns=['method', 'window_id', 'annualized_return', 'annualized_volatility', 
                                         'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'calmar_ratio'])
        empty_df.to_csv(results_dir / "backtest_results.csv", index=False)
        print("Warning: No backtest results to save")
    
    # Save portfolios (sample if too large)
    if len(portfolios_df) > 10000:
        portfolios_sample = portfolios_df.sample(n=10000, random_state=42)
        portfolios_sample.to_csv(results_dir / "portfolios.csv", index=False)
        print(f"Saved sample of {len(portfolios_sample)} portfolios (from {len(portfolios_df)} total)")
    else:
        portfolios_df.to_csv(results_dir / "portfolios.csv", index=False)
        print(f"Saved {len(portfolios_df)} portfolios")
    
    logger.log_artifact(str(results_dir / "portfolios.csv"))
    
    logger.end_run()
    
    print("\n" + "=" * 80)
    print("Pipeline execution completed successfully!")
    print("=" * 80)
    print(f"\nResults saved to: {results_dir}")
    print(f"Logs saved to: {logger.log_directory}")


if __name__ == "__main__":
    main()

