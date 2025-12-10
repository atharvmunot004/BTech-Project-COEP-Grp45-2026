"""
Report generation module for CVaR optimization results.

Generates comprehensive markdown reports with all CVaR optimization metrics and insights.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime


def generate_report(
    metrics_df: pd.DataFrame,
    output_path: Union[str, Path],
    cvar_settings: Optional[Dict] = None,
    report_sections: Optional[List[str]] = None
) -> str:
    """
    Generate comprehensive markdown report from metrics DataFrame.
    
    Args:
        metrics_df: DataFrame with all computed metrics
        output_path: Path to save the report
        cvar_settings: Dictionary with CVaR optimization settings
        report_sections: List of sections to include
        
    Returns:
        Report content as string
    """
    if report_sections is None:
        report_sections = [
            "methodology_overview",
            "cvar_theory_and_definition",
            "rockafellar_uryasev_linear_program_formulation",
            "scenario_generation_procedure",
            "cvar_optimization_results",
            "cvar_return_frontier_analysis",
            "comparison_with_markowitz_solutions",
            "portfolio_structure_effects",
            "backtesting_and_tail_risk_behavior",
            "robustness_and_sensitivity_checks",
            "computational_performance",
            "key_insights"
        ]
    
    report_lines = []
    
    # Header
    report_lines.append("# CVaR Portfolio Optimization Report")
    report_lines.append("")
    report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Methodology Overview
    if "methodology_overview" in report_sections:
        report_lines.append("## Methodology Overview")
        report_lines.append("")
        report_lines.append("### Conditional Value at Risk (CVaR) Optimization")
        report_lines.append("")
        report_lines.append("CVaR optimization focuses on minimizing tail risk by optimizing the expected loss beyond the Value at Risk (VaR) threshold.")
        report_lines.append("This approach is particularly effective for managing extreme downside risk.")
        report_lines.append("")
    
    # CVaR Theory
    if "cvar_theory_and_definition" in report_sections:
        report_lines.append("## CVaR Theory and Definition")
        report_lines.append("")
        report_lines.append("### Value at Risk (VaR)")
        report_lines.append("")
        report_lines.append("VaR_α(X) = -inf{x : P(X ≤ x) ≥ α}")
        report_lines.append("")
        report_lines.append("VaR represents the maximum loss that will not be exceeded with probability α.")
        report_lines.append("")
        report_lines.append("### Conditional Value at Risk (CVaR)")
        report_lines.append("")
        report_lines.append("CVaR_α(X) = E[X | X ≤ VaR_α(X)]")
        report_lines.append("")
        report_lines.append("CVaR (also known as Expected Shortfall) represents the expected loss given that the loss exceeds VaR.")
        report_lines.append("CVaR is a coherent risk measure and provides better tail risk assessment than VaR alone.")
        report_lines.append("")
    
    # Rockafellar-Uryasev Formulation
    if "rockafellar_uryasev_linear_program_formulation" in report_sections:
        report_lines.append("## Rockafellar-Uryasev Linear Programming Formulation")
        report_lines.append("")
        report_lines.append("The CVaR optimization problem is reformulated as a linear program:")
        report_lines.append("")
        report_lines.append("```")
        report_lines.append("minimize: λ_cvar * (VaR + (1/(1-α)) * Σ z_s / S) + λ_return * (-μ^T w)")
        report_lines.append("subject to:")
        report_lines.append("  z_s ≥ -portfolio_return_s - VaR  for all scenarios s")
        report_lines.append("  z_s ≥ 0  for all scenarios s")
        report_lines.append("  portfolio constraints")
        report_lines.append("```")
        report_lines.append("")
        report_lines.append("Where:")
        report_lines.append("- α = confidence level")
        report_lines.append("- S = number of scenarios")
        report_lines.append("- z_s = auxiliary variables for each scenario")
        report_lines.append("")
    
    # Scenario Generation
    if "scenario_generation_procedure" in report_sections and cvar_settings:
        report_lines.append("## Scenario Generation Procedure")
        report_lines.append("")
        scenario_settings = cvar_settings.get('scenario_generation', {})
        report_lines.append(f"- **Source:** {scenario_settings.get('source', 'historical')}")
        windows = scenario_settings.get('estimation_windows', [])
        report_lines.append(f"- **Estimation Windows:** {windows} days")
        report_lines.append(f"- **Block Bootstrap:** {scenario_settings.get('use_block_bootstrap', False)}")
        report_lines.append("")
    
    # Results
    if "cvar_optimization_results" in report_sections and len(metrics_df) > 0:
        report_lines.append("## CVaR Optimization Results")
        report_lines.append("")
        
        # Summary statistics
        if 'cvar' in metrics_df.columns:
            report_lines.append("### CVaR Statistics")
            report_lines.append("")
            report_lines.append(f"- **Mean CVaR:** {metrics_df['cvar'].mean():.6f}")
            report_lines.append(f"- **Min CVaR:** {metrics_df['cvar'].min():.6f}")
            report_lines.append(f"- **Max CVaR:** {metrics_df['cvar'].max():.6f}")
            report_lines.append("")
        
        if 'expected_return' in metrics_df.columns:
            report_lines.append("### Expected Return Statistics")
            report_lines.append("")
            report_lines.append(f"- **Mean Expected Return:** {metrics_df['expected_return'].mean():.6f}")
            report_lines.append(f"- **Min Expected Return:** {metrics_df['expected_return'].min():.6f}")
            report_lines.append(f"- **Max Expected Return:** {metrics_df['expected_return'].max():.6f}")
            report_lines.append("")
        
        if 'cvar_sharpe_ratio' in metrics_df.columns:
            report_lines.append("### CVaR-Adjusted Sharpe Ratios")
            report_lines.append("")
            report_lines.append(f"- **Mean CVaR Sharpe:** {metrics_df['cvar_sharpe_ratio'].mean():.6f}")
            report_lines.append(f"- **Max CVaR Sharpe:** {metrics_df['cvar_sharpe_ratio'].max():.6f}")
            report_lines.append("")
    
    # Frontier Analysis
    if "cvar_return_frontier_analysis" in report_sections:
        report_lines.append("## CVaR-Return Frontier Analysis")
        report_lines.append("")
        report_lines.append("The CVaR-return frontier shows the tradeoff between expected return and tail risk (CVaR).")
        report_lines.append("Portfolios on the efficient frontier minimize CVaR for a given target return.")
        report_lines.append("")
    
    # Structure Effects
    if "portfolio_structure_effects" in report_sections and len(metrics_df) > 0:
        report_lines.append("## Portfolio Structure Effects")
        report_lines.append("")
        if 'hhi_concentration' in metrics_df.columns:
            report_lines.append(f"- **Mean HHI Concentration:** {metrics_df['hhi_concentration'].mean():.4f}")
        if 'effective_number_of_assets' in metrics_df.columns:
            report_lines.append(f"- **Mean Effective Number of Assets:** {metrics_df['effective_number_of_assets'].mean():.2f}")
        report_lines.append("")
    
    # Tail Risk
    if "backtesting_and_tail_risk_behavior" in report_sections:
        report_lines.append("## Backtesting and Tail Risk Behavior")
        report_lines.append("")
        report_lines.append("CVaR optimization focuses on managing extreme tail risk.")
        report_lines.append("The optimized portfolios should demonstrate lower tail losses compared to mean-variance portfolios.")
        report_lines.append("")
    
    # Computational Performance
    if "computational_performance" in report_sections and len(metrics_df) > 0:
        report_lines.append("## Computational Performance")
        report_lines.append("")
        if 'runtime_per_optimization_ms' in metrics_df.columns:
            report_lines.append(f"- **Mean Runtime:** {metrics_df['runtime_per_optimization_ms'].mean():.2f} ms")
        if 'solver_time_ms' in metrics_df.columns:
            report_lines.append(f"- **Mean Solver Time:** {metrics_df['solver_time_ms'].mean():.2f} ms")
        report_lines.append("")
    
    # Key Insights
    if "key_insights" in report_sections:
        report_lines.append("## Key Insights")
        report_lines.append("")
        report_lines.append("1. CVaR optimization provides superior tail risk management compared to variance-based approaches.")
        report_lines.append("2. The Rockafellar-Uryasev formulation enables efficient linear programming solution.")
        report_lines.append("3. Scenario-based approach captures non-normal return distributions effectively.")
        report_lines.append("")
    
    # Write report
    report_content = "\n".join(report_lines)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return report_content

