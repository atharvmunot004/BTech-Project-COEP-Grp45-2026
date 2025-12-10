"""
Report generation module for Risk Parity ERC optimization results.

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
    rp_settings: Optional[Dict] = None,
    report_sections: Optional[List[str]] = None
) -> str:
    """
    Generate comprehensive markdown report from metrics DataFrame.
    
    Args:
        metrics_df: DataFrame with all computed metrics
        output_path: Path to save the report
        rp_settings: Dictionary with Risk Parity optimization settings
        report_sections: List of sections to include
        
    Returns:
        Report content as string
    """
    if report_sections is None:
        report_sections = [
            "methodology_overview",
            "risk_parity_theory",
            "equal_risk_contribution_framework",
            "covariance_estimation_methods",
            "optimization_procedure",
            "risk_contribution_results",
            "portfolio_quality_metrics",
            "comparison_with_baseline_portfolios",
            "robustness_and_sensitivity_analysis",
            "portfolio_structure_effects",
            "computational_performance",
            "key_insights"
        ]
    
    report_lines = []
    
    # Header
    report_lines.append("# Risk Parity / Equal Risk Contribution (ERC) Portfolio Optimization Report")
    report_lines.append("")
    report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Methodology Overview
    if "methodology_overview" in report_sections:
        report_lines.append("## Methodology Overview")
        report_lines.append("")
        report_lines.append("Risk Parity (Equal Risk Contribution) portfolio optimization aims to equalize the risk contribution of each asset in the portfolio.")
        report_lines.append("Unlike traditional mean-variance optimization, ERC focuses on risk diversification rather than return maximization.")
        report_lines.append("")
    
    # Risk Parity Theory
    if "risk_parity_theory" in report_sections:
        report_lines.append("## Risk Parity Theory")
        report_lines.append("")
        report_lines.append("### Key Concepts")
        report_lines.append("")
        report_lines.append("1. **Risk Contribution:** The contribution of each asset to portfolio risk")
        report_lines.append("   - RC_i = w_i * (Σ * w)_i / σ_p")
        report_lines.append("   - Where w_i is weight, Σ is covariance matrix, σ_p is portfolio volatility")
        report_lines.append("")
        report_lines.append("2. **Equal Risk Contribution:** Each asset contributes equally to portfolio risk")
        report_lines.append("   - RC_i = RC_j for all assets i, j")
        report_lines.append("   - Target: RC_i = σ_p / n (where n is number of assets)")
        report_lines.append("")
        report_lines.append("3. **Benefits:**")
        report_lines.append("   - Better risk diversification")
        report_lines.append("   - Reduced concentration risk")
        report_lines.append("   - More stable portfolio structure")
        report_lines.append("")
    
    # Equal Risk Contribution Framework
    if "equal_risk_contribution_framework" in report_sections:
        report_lines.append("## Equal Risk Contribution Framework")
        report_lines.append("")
        report_lines.append("### Optimization Objective")
        report_lines.append("")
        report_lines.append("Minimize: sum((RC_i - target_RC)^2)")
        report_lines.append("")
        report_lines.append("Where target_RC = σ_p / n (equal risk contribution per asset)")
        report_lines.append("")
        if rp_settings:
            gen_rp = rp_settings.get('generalized_rp', {})
            report_lines.append(f"- **Generalized RP Enabled:** {gen_rp.get('enable', True)}")
            report_lines.append(f"- **Target Risk Contributions:** {gen_rp.get('target_risk_contributions', 'equal')}")
        report_lines.append("")
    
    # Covariance Estimation Methods
    if "covariance_estimation_methods" in report_sections:
        report_lines.append("## Covariance Estimation Methods")
        report_lines.append("")
        if rp_settings:
            cov_settings = rp_settings.get('covariance_estimation', {})
            report_lines.append(f"- **Method:** {cov_settings.get('method', 'sample')}")
            report_lines.append(f"- **Estimation Windows:** {cov_settings.get('estimation_windows', [252, 500, 750])}")
            shrinkage = cov_settings.get('shrinkage', {})
            report_lines.append(f"- **Use Shrinkage:** {shrinkage.get('use_shrinkage', False)}")
            if shrinkage.get('use_shrinkage', False):
                report_lines.append(f"- **Shrinkage Method:** {shrinkage.get('method', 'ledoit_wolf')}")
        report_lines.append("")
    
    # Optimization Procedure
    if "optimization_procedure" in report_sections:
        report_lines.append("## Optimization Procedure")
        report_lines.append("")
        if rp_settings:
            opt_settings = rp_settings.get('optimization', {})
            report_lines.append(f"- **Solver:** {opt_settings.get('solver', 'nonlinear_programming')}")
            report_lines.append(f"- **Backend:** {opt_settings.get('backend', 'scipy_slsqp')}")
            report_lines.append(f"- **Tolerance:** {opt_settings.get('tolerance', 1e-8)}")
            report_lines.append(f"- **Max Iterations:** {opt_settings.get('max_iterations', 5000)}")
            
            constraints = rp_settings.get('constraints', {})
            report_lines.append("")
            report_lines.append("### Constraints")
            report_lines.append(f"- **Long Only:** {constraints.get('long_only', True)}")
            report_lines.append(f"- **Fully Invested:** {constraints.get('fully_invested', True)}")
            report_lines.append(f"- **Weight Bounds:** {constraints.get('weight_bounds', [0.0, 1.0])}")
            report_lines.append(f"- **Max Weight Per Asset:** {constraints.get('max_weight_per_asset', 0.25)}")
        report_lines.append("")
    
    # Risk Contribution Results
    if "risk_contribution_results" in report_sections:
        report_lines.append("## Risk Contribution Results")
        report_lines.append("")
        
        if 'risk_parity_deviation_score' in metrics_df.columns:
            avg_deviation = metrics_df['risk_parity_deviation_score'].mean()
            report_lines.append(f"- **Average Risk Parity Deviation Score:** {avg_deviation:.6f}")
            report_lines.append("  - Lower values indicate better equal risk contribution")
        
        if 'risk_contribution_coefficient_of_variation' in metrics_df.columns:
            avg_cv = metrics_df['risk_contribution_coefficient_of_variation'].mean()
            report_lines.append(f"- **Average Risk Contribution CV:** {avg_cv:.6f}")
            report_lines.append("  - Lower values indicate more equal risk contributions")
        
        if 'equal_risk_gap' in metrics_df.columns:
            avg_gap = metrics_df['equal_risk_gap'].mean()
            report_lines.append(f"- **Average Equal Risk Gap:** {avg_gap:.6f}")
        
        if 'max_risk_contribution' in metrics_df.columns and 'min_risk_contribution' in metrics_df.columns:
            avg_max = metrics_df['max_risk_contribution'].mean()
            avg_min = metrics_df['min_risk_contribution'].mean()
            report_lines.append(f"- **Average Max Risk Contribution:** {avg_max:.6f}")
            report_lines.append(f"- **Average Min Risk Contribution:** {avg_min:.6f}")
            report_lines.append(f"- **Average Range:** {avg_max - avg_min:.6f}")
        
        report_lines.append("")
    
    # Portfolio Quality Metrics
    if "portfolio_quality_metrics" in report_sections:
        report_lines.append("## Portfolio Quality Metrics")
        report_lines.append("")
        
        if 'expected_return' in metrics_df.columns:
            avg_return = metrics_df['expected_return'].mean() * 252  # Annualized
            report_lines.append(f"- **Average Expected Return (Annualized):** {avg_return:.4f}")
        
        if 'volatility' in metrics_df.columns:
            avg_vol = metrics_df['volatility'].mean() * np.sqrt(252)  # Annualized
            report_lines.append(f"- **Average Volatility (Annualized):** {avg_vol:.4f}")
        
        if 'sharpe_ratio' in metrics_df.columns:
            avg_sharpe = metrics_df['sharpe_ratio'].mean()
            max_sharpe = metrics_df['sharpe_ratio'].max()
            report_lines.append(f"- **Average Sharpe Ratio:** {avg_sharpe:.4f}")
            report_lines.append(f"- **Maximum Sharpe Ratio:** {max_sharpe:.4f}")
        
        if 'sortino_ratio' in metrics_df.columns:
            avg_sortino = metrics_df['sortino_ratio'].mean()
            report_lines.append(f"- **Average Sortino Ratio:** {avg_sortino:.4f}")
        
        if 'max_drawdown' in metrics_df.columns:
            avg_dd = metrics_df['max_drawdown'].mean()
            report_lines.append(f"- **Average Max Drawdown:** {avg_dd:.4f}")
        
        if 'calmar_ratio' in metrics_df.columns:
            avg_calmar = metrics_df['calmar_ratio'].mean()
            report_lines.append(f"- **Average Calmar Ratio:** {avg_calmar:.4f}")
        
        report_lines.append("")
    
    # Comparison with Baseline Portfolios
    if "comparison_with_baseline_portfolios" in report_sections:
        report_lines.append("## Comparison with Baseline Portfolios")
        report_lines.append("")
        
        if 'volatility_reduction_vs_baseline' in metrics_df.columns:
            avg_reduction = metrics_df['volatility_reduction_vs_baseline'].mean()
            if not np.isnan(avg_reduction):
                report_lines.append(f"- **Average Volatility Reduction vs Baseline:** {avg_reduction:.4f} ({avg_reduction*100:.2f}%)")
        
        if 'sharpe_improvement_vs_baseline' in metrics_df.columns:
            avg_improvement = metrics_df['sharpe_improvement_vs_baseline'].mean()
            if not np.isnan(avg_improvement):
                report_lines.append(f"- **Average Sharpe Improvement vs Baseline:** {avg_improvement:.4f}")
        
        if 'erc_vs_equal_weight_volatility' in metrics_df.columns:
            avg_diff = metrics_df['erc_vs_equal_weight_volatility'].mean()
            if not np.isnan(avg_diff):
                report_lines.append(f"- **ERC vs Equal Weight Volatility Difference:** {avg_diff:.6f}")
        
        if 'erc_vs_equal_weight_sharpe' in metrics_df.columns:
            avg_diff = metrics_df['erc_vs_equal_weight_sharpe'].mean()
            if not np.isnan(avg_diff):
                report_lines.append(f"- **ERC vs Equal Weight Sharpe Difference:** {avg_diff:.4f}")
        
        report_lines.append("")
    
    # Robustness and Sensitivity Analysis
    if "robustness_and_sensitivity_analysis" in report_sections:
        report_lines.append("## Robustness and Sensitivity Analysis")
        report_lines.append("")
        report_lines.append("### Covariance Window Sensitivity")
        report_lines.append("")
        report_lines.append("Different estimation windows were tested to assess robustness of ERC portfolios.")
        report_lines.append("")
        
        if 'covariance_estimation_time_ms' in metrics_df.columns:
            avg_cov_time = metrics_df['covariance_estimation_time_ms'].mean()
            report_lines.append(f"- **Average Covariance Estimation Time:** {avg_cov_time:.2f} ms")
        
        report_lines.append("")
    
    # Portfolio Structure Effects
    if "portfolio_structure_effects" in report_sections:
        report_lines.append("## Portfolio Structure Effects")
        report_lines.append("")
        
        if 'num_assets_in_portfolio' in metrics_df.columns:
            avg_assets = metrics_df['num_assets_in_portfolio'].mean()
            report_lines.append(f"- **Average Number of Assets:** {avg_assets:.1f}")
        
        if 'hhi_concentration' in metrics_df.columns:
            avg_hhi = metrics_df['hhi_concentration'].mean()
            report_lines.append(f"- **Average HHI Concentration:** {avg_hhi:.4f}")
            report_lines.append("  - Lower values indicate less concentration")
        
        if 'effective_number_of_assets' in metrics_df.columns:
            avg_ena = metrics_df['effective_number_of_assets'].mean()
            report_lines.append(f"- **Average Effective Number of Assets:** {avg_ena:.2f}")
        
        if 'weight_entropy' in metrics_df.columns:
            avg_entropy = metrics_df['weight_entropy'].mean()
            report_lines.append(f"- **Average Weight Entropy:** {avg_entropy:.4f}")
            report_lines.append("  - Higher values indicate more diversification")
        
        report_lines.append("")
    
    # Computational Performance
    if "computational_performance" in report_sections:
        report_lines.append("## Computational Performance")
        report_lines.append("")
        
        if 'runtime_per_optimization_ms' in metrics_df.columns:
            avg_runtime = metrics_df['runtime_per_optimization_ms'].mean()
            p95_runtime = metrics_df['runtime_per_optimization_ms'].quantile(0.95)
            report_lines.append(f"- **Average Runtime per Optimization:** {avg_runtime:.2f} ms")
            report_lines.append(f"- **95th Percentile Runtime:** {p95_runtime:.2f} ms")
        
        if 'solver_time_ms' in metrics_df.columns:
            avg_solver = metrics_df['solver_time_ms'].mean()
            report_lines.append(f"- **Average Solver Time:** {avg_solver:.2f} ms")
        
        if 'risk_contribution_calculation_time_ms' in metrics_df.columns:
            avg_rc_time = metrics_df['risk_contribution_calculation_time_ms'].mean()
            report_lines.append(f"- **Average Risk Contribution Calculation Time:** {avg_rc_time:.2f} ms")
        
        report_lines.append("")
    
    # Key Insights
    if "key_insights" in report_sections:
        report_lines.append("## Key Insights")
        report_lines.append("")
        
        insights = []
        
        if 'risk_parity_deviation_score' in metrics_df.columns:
            avg_dev = metrics_df['risk_parity_deviation_score'].mean()
            if avg_dev < 0.1:
                insights.append("✓ ERC portfolios achieve excellent risk parity (low deviation score)")
            elif avg_dev < 0.2:
                insights.append("✓ ERC portfolios achieve good risk parity")
            else:
                insights.append("⚠ ERC portfolios show room for improvement in risk parity")
        
        if 'sharpe_ratio' in metrics_df.columns and 'volatility' in metrics_df.columns:
            avg_sharpe = metrics_df['sharpe_ratio'].mean()
            avg_vol = metrics_df['volatility'].mean() * np.sqrt(252)
            if avg_sharpe > 1.0:
                insights.append(f"✓ ERC portfolios achieve strong risk-adjusted returns (Sharpe: {avg_sharpe:.2f})")
            if avg_vol < 0.15:
                insights.append(f"✓ ERC portfolios maintain low volatility ({avg_vol:.2%})")
        
        if 'effective_number_of_assets' in metrics_df.columns:
            avg_ena = metrics_df['effective_number_of_assets'].mean()
            if avg_ena > 5:
                insights.append(f"✓ ERC portfolios show good diversification ({avg_ena:.1f} effective assets)")
        
        if not insights:
            insights.append("Review individual portfolio metrics for detailed insights.")
        
        for insight in insights:
            report_lines.append(insight)
        
        report_lines.append("")
    
    # Write report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report_content = "\n".join(report_lines)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return report_content

