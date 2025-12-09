"""
Report generation module for VaR evaluation results.

Generates comprehensive markdown reports with all evaluation metrics and insights.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime


def generate_report(
    metrics_df: pd.DataFrame,
    output_path: Union[str, Path],
    var_settings: Optional[Dict] = None,
    report_sections: Optional[List[str]] = None
) -> str:
    """
    Generate comprehensive markdown report from metrics DataFrame.
    
    Args:
        metrics_df: DataFrame with all computed metrics
        output_path: Path to save the report
        var_settings: Dictionary with VaR settings
        report_sections: List of sections to include
        
    Returns:
        Report content as string
    """
    if report_sections is None:
        report_sections = [
            "methodology_overview",
            "backtesting_results",
            "tail_risk_analysis",
            "portfolio_structure_effects",
            "robustness_and_normality_checks",
            "computational_performance",
            "key_insights"
        ]
    
    report_lines = []
    
    # Header
    report_lines.append("# Variance-Covariance Value-at-Risk Evaluation Report")
    report_lines.append("")
    report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Methodology Overview
    if "methodology_overview" in report_sections:
        report_lines.append("## Methodology Overview")
        report_lines.append("")
        report_lines.append("### Variance-Covariance (Parametric) VaR")
        report_lines.append("")
        report_lines.append("The Variance-Covariance method assumes that returns follow a normal distribution.")
        report_lines.append("VaR is calculated as:")
        report_lines.append("")
        report_lines.append("```")
        report_lines.append("VaR = -μ - z_α * σ * √(horizon)")
        report_lines.append("```")
        report_lines.append("")
        report_lines.append("Where:")
        report_lines.append("- μ = mean return")
        report_lines.append("- σ = standard deviation of returns")
        report_lines.append("- z_α = z-score for confidence level α")
        report_lines.append("- horizon = time horizon in days")
        report_lines.append("")
        
        if var_settings:
            report_lines.append("### VaR Settings")
            report_lines.append("")
            report_lines.append(f"- **Confidence Levels:** {var_settings.get('confidence_levels', [])}")
            report_lines.append(f"- **Horizons:** {var_settings.get('horizons', [])} days")
            report_lines.append(f"- **Estimation Windows:** {var_settings.get('estimation_windows', [])} days")
            report_lines.append("")
    
    # Summary Statistics
    report_lines.append("## Summary Statistics")
    report_lines.append("")
    report_lines.append(f"- **Total Portfolios Evaluated:** {len(metrics_df)}")
    report_lines.append("")
    
    # Backtesting Results
    if "backtesting_results" in report_sections:
        report_lines.append("## Backtesting Results")
        report_lines.append("")
        
        if 'hit_rate' in metrics_df.columns:
            avg_hit_rate = metrics_df['hit_rate'].mean()
            report_lines.append(f"- **Average Hit Rate:** {avg_hit_rate:.4f}")
            report_lines.append("")
        
        if 'violation_ratio' in metrics_df.columns:
            avg_violation_ratio = metrics_df['violation_ratio'].mean()
            report_lines.append(f"- **Average Violation Ratio:** {avg_violation_ratio:.4f}")
            report_lines.append("  - Ratio > 1 indicates overestimation of risk")
            report_lines.append("  - Ratio < 1 indicates underestimation of risk")
            report_lines.append("")
        
        if 'traffic_light_zone' in metrics_df.columns:
            zone_counts = metrics_df['traffic_light_zone'].value_counts()
            report_lines.append("### Traffic Light Zones")
            report_lines.append("")
            for zone, count in zone_counts.items():
                report_lines.append(f"- **{zone.capitalize()}:** {count} portfolios ({count/len(metrics_df)*100:.1f}%)")
            report_lines.append("")
        
        if 'kupiec_unconditional_coverage' in metrics_df.columns:
            kupiec_passed = (metrics_df['kupiec_unconditional_coverage'] > 0.05).sum()
            report_lines.append(f"- **Kupiec Test Passed:** {kupiec_passed}/{len(metrics_df)} portfolios ({kupiec_passed/len(metrics_df)*100:.1f}%)")
            report_lines.append("")
    
    # Tail Risk Analysis
    if "tail_risk_analysis" in report_sections:
        report_lines.append("## Tail Risk Analysis")
        report_lines.append("")
        
        if 'mean_exceedance' in metrics_df.columns:
            avg_mean_exceedance = metrics_df['mean_exceedance'].mean()
            report_lines.append(f"- **Average Mean Exceedance:** {avg_mean_exceedance:.6f}")
            report_lines.append("")
        
        if 'max_exceedance' in metrics_df.columns:
            avg_max_exceedance = metrics_df['max_exceedance'].mean()
            report_lines.append(f"- **Average Max Exceedance:** {avg_max_exceedance:.6f}")
            report_lines.append("")
    
    # Portfolio Structure Effects
    if "portfolio_structure_effects" in report_sections:
        report_lines.append("## Portfolio Structure Effects")
        report_lines.append("")
        
        if 'num_active_assets' in metrics_df.columns:
            avg_active_assets = metrics_df['num_active_assets'].mean()
            report_lines.append(f"- **Average Number of Active Assets:** {avg_active_assets:.2f}")
            report_lines.append("")
        
        if 'hhi_concentration' in metrics_df.columns:
            avg_hhi = metrics_df['hhi_concentration'].mean()
            report_lines.append(f"- **Average HHI Concentration:** {avg_hhi:.4f}")
            report_lines.append("  - Higher values indicate more concentrated portfolios")
            report_lines.append("")
        
        if 'effective_number_of_assets' in metrics_df.columns:
            avg_enc = metrics_df['effective_number_of_assets'].mean()
            report_lines.append(f"- **Average Effective Number of Assets:** {avg_enc:.2f}")
            report_lines.append("")
    
    # Robustness and Normality Checks
    if "robustness_and_normality_checks" in report_sections:
        report_lines.append("## Robustness and Normality Checks")
        report_lines.append("")
        
        if 'skewness' in metrics_df.columns:
            avg_skewness = metrics_df['skewness'].mean()
            report_lines.append(f"- **Average Skewness:** {avg_skewness:.4f}")
            report_lines.append("  - Negative values indicate left-skewed (tail risk)")
            report_lines.append("")
        
        if 'kurtosis' in metrics_df.columns:
            avg_kurtosis = metrics_df['kurtosis'].mean()
            report_lines.append(f"- **Average Excess Kurtosis:** {avg_kurtosis:.4f}")
            report_lines.append("  - Positive values indicate fat tails")
            report_lines.append("")
        
        if 'jarque_bera_p_value' in metrics_df.columns:
            normality_passed = (metrics_df['jarque_bera_p_value'] > 0.05).sum()
            report_lines.append(f"- **Normality Tests Passed (Jarque-Bera):** {normality_passed}/{len(metrics_df)} portfolios ({normality_passed/len(metrics_df)*100:.1f}%)")
            report_lines.append("")
    
    # Computational Performance
    if "computational_performance" in report_sections:
        report_lines.append("## Computational Performance")
        report_lines.append("")
        
        if 'runtime_per_portfolio_ms' in metrics_df.columns:
            avg_runtime = metrics_df['runtime_per_portfolio_ms'].mean()
            report_lines.append(f"- **Average Runtime per Portfolio:** {avg_runtime:.2f} ms")
            report_lines.append("")
        
        if 'p95_runtime_ms' in metrics_df.columns:
            p95_runtime = metrics_df['p95_runtime_ms'].mean()
            report_lines.append(f"- **95th Percentile Runtime:** {p95_runtime:.2f} ms")
            report_lines.append("")
    
    # Key Insights
    if "key_insights" in report_sections:
        report_lines.append("## Key Insights")
        report_lines.append("")
        report_lines.append("### Findings")
        report_lines.append("")
        
        if 'violation_ratio' in metrics_df.columns:
            avg_vr = metrics_df['violation_ratio'].mean()
            if avg_vr > 1.2:
                report_lines.append("- **Risk Overestimation:** VaR tends to overestimate risk (violation ratio > 1.2)")
            elif avg_vr < 0.8:
                report_lines.append("- **Risk Underestimation:** VaR tends to underestimate risk (violation ratio < 0.8)")
            else:
                report_lines.append("- **Adequate Risk Estimation:** VaR provides reasonable risk estimates")
            report_lines.append("")
        
        if 'kurtosis' in metrics_df.columns:
            avg_kurt = metrics_df['kurtosis'].mean()
            if avg_kurt > 3:
                report_lines.append("- **Fat Tails Detected:** Returns exhibit fat tails, which may limit VaR accuracy")
            report_lines.append("")
        
        report_lines.append("### Recommendations")
        report_lines.append("")
        report_lines.append("- Consider alternative VaR methods (e.g., Monte Carlo, Historical Simulation) for portfolios with fat-tailed returns")
        report_lines.append("- Monitor portfolios in 'red' traffic light zone more closely")
        report_lines.append("- Adjust confidence levels or horizons based on backtesting results")
        report_lines.append("")
    
    # Detailed Metrics Table
    report_lines.append("## Detailed Metrics")
    report_lines.append("")
    report_lines.append("### Summary Statistics by Metric")
    report_lines.append("")
    
    numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
    summary_stats = metrics_df[numeric_cols].describe()
    
    report_lines.append("```")
    report_lines.append(summary_stats.to_string())
    report_lines.append("```")
    report_lines.append("")
    
    report_content = "\n".join(report_lines)
    
    # Save report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return report_content

