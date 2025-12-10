"""
Report generation module for Black-Litterman optimization results.

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
    bl_settings: Optional[Dict] = None,
    report_sections: Optional[List[str]] = None
) -> str:
    """
    Generate comprehensive markdown report from metrics DataFrame.
    
    Args:
        metrics_df: DataFrame with all computed metrics
        output_path: Path to save the report
        bl_settings: Dictionary with Black-Litterman optimization settings
        report_sections: List of sections to include
        
    Returns:
        Report content as string
    """
    if report_sections is None:
        report_sections = [
            "methodology_overview",
            "black_litterman_theory",
            "equilibrium_prior_estimation_from_portfolios",
            "synthetic_view_generation_method",
            "posterior_distribution_derivation",
            "optimization_results",
            "efficient_frontier_analysis",
            "comparison_with_markowitz_results",
            "impact_of_views_and_confidence_levels",
            "robustness_and_sensitivity_analysis",
            "portfolio_structure_effects",
            "computational_performance",
            "key_insights"
        ]
    
    report_lines = []
    
    # Header
    report_lines.append("# Black-Litterman Portfolio Optimization Report")
    report_lines.append("")
    report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Methodology Overview
    if "methodology_overview" in report_sections:
        report_lines.append("## Methodology Overview")
        report_lines.append("")
        report_lines.append("The Black-Litterman model combines market equilibrium returns (prior) with investor views to produce posterior expected returns.")
        report_lines.append("This approach addresses the instability of mean-variance optimization by using market-implied returns as a starting point.")
        report_lines.append("")
    
    # Black-Litterman Theory
    if "black_litterman_theory" in report_sections:
        report_lines.append("## Black-Litterman Theory")
        report_lines.append("")
        report_lines.append("### Key Components")
        report_lines.append("")
        report_lines.append("1. **Prior Returns (π):** Market equilibrium returns derived from baseline portfolios")
        report_lines.append("   - π = λ * Σ * w_market")
        report_lines.append("   - Where λ is risk aversion, Σ is covariance, w_market is portfolio weights")
        report_lines.append("")
        report_lines.append("2. **Synthetic Investor Views:**")
        report_lines.append("   - Generated from return differentials between assets")
        report_lines.append("   - Relative views: Expected return differences between asset pairs")
        report_lines.append("")
        report_lines.append("3. **Posterior Returns (μ_BL):**")
        report_lines.append("   - μ_BL = [(τΣ)^(-1) + P^T * Ω^(-1) * P]^(-1) * [(τΣ)^(-1) * π + P^T * Ω^(-1) * Q]")
        report_lines.append("   - Where P is pick matrix, Q is view vector, Ω is uncertainty matrix, τ is scaling factor")
        report_lines.append("")
    
    # Equilibrium Prior Estimation from Portfolios
    if "equilibrium_prior_estimation_from_portfolios" in report_sections:
        report_lines.append("## Equilibrium Prior Estimation from Portfolios")
        report_lines.append("")
        if bl_settings:
            market_eq = bl_settings.get('market_equilibrium', {})
            report_lines.append(f"- **Derive from Portfolios:** {market_eq.get('derive_pi_from_portfolios', True)}")
            report_lines.append(f"- **Market Weight Source:** {market_eq.get('market_weight_source', 'baseline_portfolios')}")
            report_lines.append(f"- **Risk Aversion (λ):** {bl_settings.get('risk_aversion', 2.5)}")
        report_lines.append("")
    
    # Synthetic View Generation Method
    if "synthetic_view_generation_method" in report_sections:
        report_lines.append("## Synthetic View Generation Method")
        report_lines.append("")
        if bl_settings:
            views = bl_settings.get('views', {})
            report_lines.append(f"- **Generate Synthetic Views:** {views.get('generate_synthetic_views', True)}")
            report_lines.append(f"- **View Generation Method:** {views.get('view_generation_method', 'return_differentials')}")
            report_lines.append(f"- **Number of Views:** {views.get('num_views', 5)}")
            report_lines.append(f"- **Uncertainty Matrix:** {views.get('uncertainty_matrix', 'diagonal')}")
            report_lines.append(f"- **Tau (τ):** {bl_settings.get('tau', 0.025)}")
        report_lines.append("")
    
    # Posterior Distribution Derivation
    if "posterior_distribution_derivation" in report_sections:
        report_lines.append("## Posterior Distribution Derivation")
        report_lines.append("")
        report_lines.append("The posterior distribution combines prior beliefs with synthetic investor views using Bayesian updating.")
        report_lines.append("The posterior covariance matrix accounts for the reduction in uncertainty from incorporating views.")
        report_lines.append("")
    
    # Optimization Results
    if "optimization_results" in report_sections:
        report_lines.append("## Optimization Results")
        report_lines.append("")
        
        if len(metrics_df) > 0:
            row = metrics_df.iloc[0]
            
            report_lines.append("### Portfolio Performance Metrics")
            report_lines.append("")
            
            if 'expected_return' in row:
                report_lines.append(f"- **Expected Return:** {row['expected_return']:.4f} (annualized)")
            if 'volatility' in row:
                report_lines.append(f"- **Volatility:** {row['volatility']:.4f} (annualized)")
            if 'sharpe_ratio' in row:
                report_lines.append(f"- **Sharpe Ratio:** {row['sharpe_ratio']:.4f}")
            if 'sortino_ratio' in row:
                report_lines.append(f"- **Sortino Ratio:** {row['sortino_ratio']:.4f}")
            if 'max_drawdown' in row:
                report_lines.append(f"- **Maximum Drawdown:** {row['max_drawdown']:.4f}")
            if 'calmar_ratio' in row:
                report_lines.append(f"- **Calmar Ratio:** {row['calmar_ratio']:.4f}")
            
            report_lines.append("")
            
            report_lines.append("### Black-Litterman Specific Metrics")
            report_lines.append("")
            
            if 'prior_vs_posterior_distance' in row:
                report_lines.append(f"- **Prior vs Posterior Distance:** {row['prior_vs_posterior_distance']:.4f}")
            if 'prior_vs_posterior_correlation' in row:
                report_lines.append(f"- **Prior vs Posterior Correlation:** {row['prior_vs_posterior_correlation']:.4f}")
            if 'view_consistency' in row:
                report_lines.append(f"- **View Consistency:** {row['view_consistency']:.4f}")
            if 'view_impact_magnitude' in row:
                report_lines.append(f"- **View Impact Magnitude:** {row['view_impact_magnitude']:.4f}")
            if 'posterior_sharpe_vs_prior_sharpe' in row:
                report_lines.append(f"- **Sharpe Improvement:** {row['posterior_sharpe_vs_prior_sharpe']:.4f}")
            if 'information_gain_from_views' in row:
                report_lines.append(f"- **Information Gain from Views:** {row['information_gain_from_views']:.4f}")
            
            report_lines.append("")
            
            report_lines.append("### Market Comparison Metrics")
            report_lines.append("")
            
            if 'information_ratio' in row:
                report_lines.append(f"- **Information Ratio:** {row['information_ratio']:.4f}")
            if 'tracking_error_vs_market' in row:
                report_lines.append(f"- **Tracking Error vs Market:** {row['tracking_error_vs_market']:.4f}")
            if 'alpha_vs_market_portfolio' in row:
                report_lines.append(f"- **Alpha vs Market:** {row['alpha_vs_market_portfolio']:.4f}")
            
            report_lines.append("")
    
    # Efficient Frontier Analysis
    if "efficient_frontier_analysis" in report_sections:
        report_lines.append("## Efficient Frontier Analysis")
        report_lines.append("")
        report_lines.append("The efficient frontier shows the set of optimal portfolios using posterior returns and covariance.")
        report_lines.append("Each point on the frontier represents a portfolio that maximizes expected return for a given level of risk.")
        report_lines.append("")
    
    # Comparison with Markowitz
    if "comparison_with_markowitz_results" in report_sections:
        report_lines.append("## Comparison with Markowitz Results")
        report_lines.append("")
        report_lines.append("Black-Litterman portfolios typically show:")
        report_lines.append("- More stable weights (less extreme positions)")
        report_lines.append("- Better out-of-sample performance")
        report_lines.append("- Incorporation of market equilibrium as anchor")
        report_lines.append("")
    
    # Impact of Views and Confidence Levels
    if "impact_of_views_and_confidence_levels" in report_sections:
        report_lines.append("## Impact of Views and Confidence Levels")
        report_lines.append("")
        report_lines.append("Synthetic views generated from return differentials provide additional information beyond market equilibrium.")
        report_lines.append("The uncertainty matrix (Ω) controls how much views influence the posterior distribution.")
        report_lines.append("")
    
    # Robustness and Sensitivity Analysis
    if "robustness_and_sensitivity_analysis" in report_sections:
        report_lines.append("## Robustness and Sensitivity Analysis")
        report_lines.append("")
        report_lines.append("Key parameters to analyze:")
        report_lines.append("- **Tau (τ):** Controls scaling of uncertainty")
        report_lines.append("- **Risk Aversion (λ):** Affects market equilibrium returns")
        report_lines.append("- **Number of Views:** Impact on posterior distribution")
        report_lines.append("")
    
    # Portfolio Structure Effects
    if "portfolio_structure_effects" in report_sections:
        report_lines.append("## Portfolio Structure Effects")
        report_lines.append("")
        
        if len(metrics_df) > 0:
            row = metrics_df.iloc[0]
            
            if 'num_assets_in_portfolio' in row:
                report_lines.append(f"- **Number of Assets:** {row['num_assets_in_portfolio']}")
            if 'hhi_concentration' in row:
                report_lines.append(f"- **HHI Concentration:** {row['hhi_concentration']:.4f}")
            if 'effective_number_of_assets' in row:
                report_lines.append(f"- **Effective Number of Assets:** {row['effective_number_of_assets']:.2f}")
            if 'weight_entropy' in row:
                report_lines.append(f"- **Weight Entropy:** {row['weight_entropy']:.4f}")
            if 'active_share' in row:
                report_lines.append(f"- **Active Share:** {row['active_share']:.4f}")
        
        report_lines.append("")
    
    # Computational Performance
    if "computational_performance" in report_sections:
        report_lines.append("## Computational Performance")
        report_lines.append("")
        
        if len(metrics_df) > 0:
            row = metrics_df.iloc[0]
            
            if 'runtime_per_optimization_ms' in row:
                report_lines.append(f"- **Total Runtime:** {row['runtime_per_optimization_ms']:.2f} ms")
            if 'covariance_estimation_time_ms' in row:
                report_lines.append(f"- **Covariance Estimation:** {row['covariance_estimation_time_ms']:.2f} ms")
            if 'equilibrium_returns_calculation_time_ms' in row:
                report_lines.append(f"- **Equilibrium Returns:** {row['equilibrium_returns_calculation_time_ms']:.2f} ms")
            if 'view_processing_time_ms' in row:
                report_lines.append(f"- **View Processing:** {row['view_processing_time_ms']:.2f} ms")
            if 'posterior_calculation_time_ms' in row:
                report_lines.append(f"- **Posterior Calculation:** {row['posterior_calculation_time_ms']:.2f} ms")
            if 'solver_time_ms' in row:
                report_lines.append(f"- **Solver Time:** {row['solver_time_ms']:.2f} ms")
        
        report_lines.append("")
    
    # Key Insights
    if "key_insights" in report_sections:
        report_lines.append("## Key Insights")
        report_lines.append("")
        report_lines.append("### Summary")
        report_lines.append("")
        report_lines.append("The Black-Litterman model successfully combines:")
        report_lines.append("1. Market equilibrium returns derived from baseline portfolios as a stable prior")
        report_lines.append("2. Synthetic views generated from return differentials to incorporate additional insights")
        report_lines.append("3. Bayesian updating to derive posterior distributions")
        report_lines.append("")
        report_lines.append("This approach provides more stable and realistic portfolio allocations compared to traditional mean-variance optimization.")
        report_lines.append("")
    
    # Write report
    report_content = "\n".join(report_lines)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return report_content

