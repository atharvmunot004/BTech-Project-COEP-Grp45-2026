"""
Report generation module for Markowitz optimization results.

Generates comprehensive markdown reports with all optimization metrics and insights.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime


def generate_report(
    metrics_df: pd.DataFrame,
    output_path: Union[str, Path],
    markowitz_settings: Optional[Dict] = None,
    report_sections: Optional[List[str]] = None
) -> str:
    """
    Generate comprehensive markdown report from metrics DataFrame.
    
    Args:
        metrics_df: DataFrame with all computed metrics
        output_path: Path to save the report
        markowitz_settings: Dictionary with Markowitz optimization settings
        report_sections: List of sections to include
        
    Returns:
        Report content as string
    """
    if report_sections is None:
        report_sections = [
            "methodology_overview",
            "mean_variance_theory",
            "covariance_estimation",
            "efficient_frontier_construction",
            "portfolio_optimization_results",
            "risk_return_tradeoff_analysis",
            "portfolio_structure_effects",
            "sensitivity_and_constraint_analysis",
            "robustness_checks",
            "computational_performance",
            "key_insights"
        ]
    
    report_lines = []
    
    # Header
    report_lines.append("# Markowitz Mean-Variance Portfolio Optimization Report")
    report_lines.append("")
    report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Methodology Overview
    if "methodology_overview" in report_sections:
        report_lines.append("## Methodology Overview")
        report_lines.append("")
        report_lines.append("### Markowitz Mean-Variance Optimization")
        report_lines.append("")
        report_lines.append("The Markowitz (1952) mean-variance framework optimizes portfolios by balancing expected return and risk (variance).")
        report_lines.append("The optimization problem is:")
        report_lines.append("")
        report_lines.append("```")
        report_lines.append("minimize: w^T Σ w - λ * μ^T w")
        report_lines.append("subject to: Σ w_i = 1, w_i >= 0 (long-only)")
        report_lines.append("```")
        report_lines.append("")
        report_lines.append("Where:")
        report_lines.append("- w = portfolio weights vector")
        report_lines.append("- Σ = covariance matrix")
        report_lines.append("- μ = expected returns vector")
        report_lines.append("- λ = risk aversion parameter")
        report_lines.append("")
        
        if markowitz_settings:
            report_lines.append("### Optimization Settings")
            report_lines.append("")
            obj = markowitz_settings.get('objective', 'min_variance')
            report_lines.append(f"- **Objective:** {obj}")
            
            if 'risk_return_tradeoff' in markowitz_settings:
                tradeoff = markowitz_settings['risk_return_tradeoff']
                if tradeoff.get('use_risk_aversion', False):
                    lambdas = tradeoff.get('lambda_values', [])
                    report_lines.append(f"- **Risk Aversion Parameters (λ):** {lambdas}")
            
            if 'constraints' in markowitz_settings:
                constraints = markowitz_settings['constraints']
                report_lines.append(f"- **Long Only:** {constraints.get('long_only', True)}")
                report_lines.append(f"- **Fully Invested:** {constraints.get('fully_invested', True)}")
                max_weight = constraints.get('max_weight_per_asset')
                if max_weight:
                    report_lines.append(f"- **Max Weight per Asset:** {max_weight}")
            
            if 'covariance_estimation' in markowitz_settings:
                cov_settings = markowitz_settings['covariance_estimation']
                report_lines.append(f"- **Covariance Method:** {cov_settings.get('method', 'sample')}")
                windows = cov_settings.get('estimation_windows', [])
                report_lines.append(f"- **Estimation Windows:** {windows} days")
                if cov_settings.get('shrinkage', {}).get('use_shrinkage', False):
                    report_lines.append(f"- **Shrinkage:** {cov_settings['shrinkage'].get('method', 'ledoit_wolf')}")
            report_lines.append("")
    
    # Mean-Variance Theory
    if "mean_variance_theory" in report_sections:
        report_lines.append("## Mean-Variance Theory")
        report_lines.append("")
        report_lines.append("The efficient frontier represents the set of portfolios that offer the highest expected return for a given level of risk, or the lowest risk for a given expected return.")
        report_lines.append("")
        report_lines.append("Key concepts:")
        report_lines.append("- **Efficient Frontier:** Optimal risk-return combinations")
        report_lines.append("- **Minimum Variance Portfolio:** Portfolio with lowest risk")
        report_lines.append("- **Maximum Sharpe Portfolio:** Portfolio with highest risk-adjusted return")
        report_lines.append("- **Risk-Return Tradeoff:** Higher returns require accepting higher risk")
        report_lines.append("")
    
    # Summary Statistics
    report_lines.append("## Summary Statistics")
    report_lines.append("")
    report_lines.append(f"- **Total Portfolios Optimized:** {len(metrics_df)}")
    report_lines.append("")
    
    # Portfolio Optimization Results
    if "portfolio_optimization_results" in report_sections:
        report_lines.append("## Portfolio Optimization Results")
        report_lines.append("")
        
        if 'expected_return' in metrics_df.columns:
            avg_return = metrics_df['expected_return'].mean()
            report_lines.append(f"- **Average Expected Return:** {avg_return:.6f} ({avg_return*100:.2f}%)")
            report_lines.append("")
        
        if 'volatility' in metrics_df.columns:
            avg_vol = metrics_df['volatility'].mean()
            report_lines.append(f"- **Average Volatility:** {avg_vol:.6f} ({avg_vol*100:.2f}%)")
            report_lines.append("")
        
        if 'portfolio_variance' in metrics_df.columns:
            avg_var = metrics_df['portfolio_variance'].mean()
            report_lines.append(f"- **Average Portfolio Variance:** {avg_var:.6f}")
            report_lines.append("")
        
        if 'sharpe_ratio' in metrics_df.columns:
            avg_sharpe = metrics_df['sharpe_ratio'].mean()
            max_sharpe = metrics_df['sharpe_ratio'].max()
            report_lines.append(f"- **Average Sharpe Ratio:** {avg_sharpe:.4f}")
            report_lines.append(f"- **Maximum Sharpe Ratio:** {max_sharpe:.4f}")
            report_lines.append("")
        
        if 'sortino_ratio' in metrics_df.columns:
            avg_sortino = metrics_df['sortino_ratio'].mean()
            report_lines.append(f"- **Average Sortino Ratio:** {avg_sortino:.4f}")
            report_lines.append("")
    
    # Risk-Return Tradeoff Analysis
    if "risk_return_tradeoff_analysis" in report_sections:
        report_lines.append("## Risk-Return Tradeoff Analysis")
        report_lines.append("")
        
        if 'expected_return' in metrics_df.columns and 'volatility' in metrics_df.columns:
            # Compute correlation
            corr = metrics_df['expected_return'].corr(metrics_df['volatility'])
            report_lines.append(f"- **Return-Volatility Correlation:** {corr:.4f}")
            report_lines.append("  - Positive correlation indicates higher returns come with higher risk")
            report_lines.append("")
        
        if 'sharpe_ratio' in metrics_df.columns:
            sharpe_stats = metrics_df['sharpe_ratio'].describe()
            report_lines.append("### Sharpe Ratio Distribution")
            report_lines.append("")
            report_lines.append(f"- **Mean:** {sharpe_stats['mean']:.4f}")
            report_lines.append(f"- **Std:** {sharpe_stats['std']:.4f}")
            report_lines.append(f"- **Min:** {sharpe_stats['min']:.4f}")
            report_lines.append(f"- **Max:** {sharpe_stats['max']:.4f}")
            report_lines.append("")
    
    # Portfolio Structure Effects
    if "portfolio_structure_effects" in report_sections:
        report_lines.append("## Portfolio Structure Effects")
        report_lines.append("")
        
        if 'num_assets_in_portfolio' in metrics_df.columns:
            avg_assets = metrics_df['num_assets_in_portfolio'].mean()
            report_lines.append(f"- **Average Number of Assets:** {avg_assets:.2f}")
            report_lines.append("")
        
        if 'hhi_concentration' in metrics_df.columns:
            avg_hhi = metrics_df['hhi_concentration'].mean()
            report_lines.append(f"- **Average HHI Concentration:** {avg_hhi:.4f}")
            report_lines.append("  - Higher values indicate more concentrated portfolios")
            report_lines.append("  - HHI ranges from 1/n (equal weights) to 1 (single asset)")
            report_lines.append("")
        
        if 'effective_number_of_assets' in metrics_df.columns:
            avg_enc = metrics_df['effective_number_of_assets'].mean()
            report_lines.append(f"- **Average Effective Number of Assets:** {avg_enc:.2f}")
            report_lines.append("")
        
        if 'weight_entropy' in metrics_df.columns:
            avg_entropy = metrics_df['weight_entropy'].mean()
            report_lines.append(f"- **Average Weight Entropy:** {avg_entropy:.4f}")
            report_lines.append("  - Higher entropy indicates better diversification")
            report_lines.append("")
        
        if 'pairwise_correlation_mean' in metrics_df.columns:
            avg_corr = metrics_df['pairwise_correlation_mean'].mean()
            report_lines.append(f"- **Average Pairwise Correlation:** {avg_corr:.4f}")
            report_lines.append("")
    
    # Risk Metrics
    if 'value_at_risk' in metrics_df.columns:
        report_lines.append("## Risk Metrics")
        report_lines.append("")
        avg_var = metrics_df['value_at_risk'].mean()
        report_lines.append(f"- **Average Value at Risk (VaR):** {avg_var:.6f}")
        
        if 'conditional_value_at_risk' in metrics_df.columns:
            avg_cvar = metrics_df['conditional_value_at_risk'].mean()
            report_lines.append(f"- **Average Conditional VaR (CVaR):** {avg_cvar:.6f}")
        report_lines.append("")
    
    # Robustness Checks
    if "robustness_checks" in report_sections:
        report_lines.append("## Robustness and Normality Checks")
        report_lines.append("")
        
        if 'skewness' in metrics_df.columns:
            avg_skew = metrics_df['skewness'].mean()
            report_lines.append(f"- **Average Skewness:** {avg_skew:.4f}")
            report_lines.append("  - Negative skewness indicates left tail risk")
            report_lines.append("")
        
        if 'kurtosis' in metrics_df.columns:
            avg_kurt = metrics_df['kurtosis'].mean()
            report_lines.append(f"- **Average Excess Kurtosis:** {avg_kurt:.4f}")
            report_lines.append("  - Positive kurtosis indicates fat tails")
            report_lines.append("")
        
        if 'jarque_bera_p_value' in metrics_df.columns:
            rejection_rate = (metrics_df['jarque_bera_p_value'] < 0.05).mean()
            report_lines.append(f"- **Normality Rejection Rate (p < 0.05):** {rejection_rate:.2%}")
            report_lines.append("  - Higher rate indicates more non-normal return distributions")
            report_lines.append("")
    
    # Computational Performance
    if "computational_performance" in report_sections:
        report_lines.append("## Computational Performance")
        report_lines.append("")
        
        if 'runtime_per_optimization_ms' in metrics_df.columns:
            avg_runtime = metrics_df['runtime_per_optimization_ms'].mean()
            report_lines.append(f"- **Average Runtime per Optimization:** {avg_runtime:.2f} ms")
            report_lines.append("")
        
        if 'p95_runtime_ms' in metrics_df.columns:
            p95_runtime = metrics_df['p95_runtime_ms'].mean()
            report_lines.append(f"- **95th Percentile Runtime:** {p95_runtime:.2f} ms")
            report_lines.append("")
        
        if 'covariance_estimation_time_ms' in metrics_df.columns:
            avg_cov_time = metrics_df['covariance_estimation_time_ms'].mean()
            report_lines.append(f"- **Average Covariance Estimation Time:** {avg_cov_time:.2f} ms")
            report_lines.append("")
        
        if 'solver_time_ms' in metrics_df.columns:
            avg_solver_time = metrics_df['solver_time_ms'].mean()
            report_lines.append(f"- **Average Solver Time:** {avg_solver_time:.2f} ms")
            report_lines.append("")
    
    # Key Insights
    if "key_insights" in report_sections:
        report_lines.append("## Key Insights")
        report_lines.append("")
        
        # Generate insights based on metrics
        insights = []
        
        if 'sharpe_ratio' in metrics_df.columns:
            max_sharpe_idx = metrics_df['sharpe_ratio'].idxmax()
            max_sharpe = metrics_df.loc[max_sharpe_idx, 'sharpe_ratio']
            insights.append(f"- Maximum Sharpe ratio achieved: {max_sharpe:.4f}")
        
        if 'hhi_concentration' in metrics_df.columns:
            avg_hhi = metrics_df['hhi_concentration'].mean()
            if avg_hhi > 0.5:
                insights.append("- Portfolios show high concentration (HHI > 0.5)")
            elif avg_hhi < 0.2:
                insights.append("- Portfolios show good diversification (HHI < 0.2)")
        
        if 'expected_return' in metrics_df.columns and 'volatility' in metrics_df.columns:
            corr = metrics_df['expected_return'].corr(metrics_df['volatility'])
            if corr > 0.7:
                insights.append("- Strong positive risk-return tradeoff observed")
        
        if len(insights) == 0:
            insights.append("- Review individual portfolio metrics for detailed analysis")
        
        for insight in insights:
            report_lines.append(insight)
        report_lines.append("")
    
    # Write report
    report_content = "\n".join(report_lines)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return report_content

