"""
Report generation module for Monte Carlo VaR/CVaR evaluation results.

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
    monte_carlo_settings: Optional[Dict] = None,
    report_sections: Optional[List[str]] = None
) -> str:
    """
    Generate comprehensive markdown report from metrics DataFrame.
    
    Args:
        metrics_df: DataFrame with all computed metrics
        output_path: Path to save the report
        monte_carlo_settings: Dictionary with Monte Carlo settings
        report_sections: List of sections to include
        
    Returns:
        Report content as string
    """
    if report_sections is None:
        report_sections = [
            "methodology_overview",
            "monte_carlo_simulation_details",
            "backtesting_results",
            "var_cvar_comparison",
            "tail_risk_analysis",
            "portfolio_structure_effects",
            "robustness_and_normality_checks",
            "computational_performance",
            "key_insights"
        ]
    
    report_lines = []
    
    # Header
    report_lines.append("# Monte Carlo Simulation for VaR/CVaR Evaluation Report")
    report_lines.append("")
    report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Methodology Overview
    if "methodology_overview" in report_sections:
        report_lines.append("## Methodology Overview")
        report_lines.append("")
        report_lines.append("### Monte Carlo Simulation for VaR and CVaR")
        report_lines.append("")
        report_lines.append("Monte Carlo simulation generates multiple scenarios of portfolio returns based on historical data.")
        report_lines.append("VaR and CVaR are then calculated from the distribution of simulated returns.")
        report_lines.append("")
        report_lines.append("**Advantages:**")
        report_lines.append("- Can capture non-normal distributions")
        report_lines.append("- Flexible with different distribution assumptions")
        report_lines.append("- Provides both VaR and CVaR estimates")
        report_lines.append("")
        report_lines.append("**Method:**")
        report_lines.append("1. Estimate mean and covariance from historical returns")
        report_lines.append("2. Simulate multiple scenarios of future returns")
        report_lines.append("3. Calculate VaR as the quantile of simulated losses")
        report_lines.append("4. Calculate CVaR as the expected loss beyond VaR")
        report_lines.append("")
    
    # Monte Carlo Simulation Details
    if "monte_carlo_simulation_details" in report_sections:
        report_lines.append("## Monte Carlo Simulation Details")
        report_lines.append("")
        
        if monte_carlo_settings:
            report_lines.append("### Simulation Parameters")
            report_lines.append("")
            report_lines.append(f"- **Number of Simulations:** {monte_carlo_settings.get('num_simulations', 'N/A'):,}")
            report_lines.append(f"- **Distribution Type:** {monte_carlo_settings.get('distribution_type', 'N/A')}")
            report_lines.append(f"- **Random Seed:** {monte_carlo_settings.get('random_seed', 'N/A')}")
            report_lines.append(f"- **Confidence Levels:** {monte_carlo_settings.get('confidence_levels', [])}")
            report_lines.append(f"- **Horizons:** {monte_carlo_settings.get('horizons', [])} days")
            report_lines.append(f"- **Estimation Windows:** {monte_carlo_settings.get('estimation_windows', [])} days")
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
    
    # VaR/CVaR Comparison
    if "var_cvar_comparison" in report_sections:
        report_lines.append("## VaR vs CVaR Comparison")
        report_lines.append("")
        report_lines.append("CVaR (Conditional Value at Risk) provides additional insight into tail risk beyond VaR.")
        report_lines.append("")
        
        if 'cvar_mean_exceedance' in metrics_df.columns:
            avg_cvar_mean = metrics_df['cvar_mean_exceedance'].mean()
            avg_var_mean = metrics_df['mean_exceedance'].mean() if 'mean_exceedance' in metrics_df.columns else np.nan
            report_lines.append(f"- **Average CVaR Mean Exceedance:** {avg_cvar_mean:.6f}")
            if not np.isnan(avg_var_mean):
                report_lines.append(f"- **Average VaR Mean Exceedance:** {avg_var_mean:.6f}")
                report_lines.append(f"- **Difference:** {avg_cvar_mean - avg_var_mean:.6f}")
            report_lines.append("")
        
        if 'cvar_max_exceedance' in metrics_df.columns:
            avg_cvar_max = metrics_df['cvar_max_exceedance'].mean()
            avg_var_max = metrics_df['max_exceedance'].mean() if 'max_exceedance' in metrics_df.columns else np.nan
            report_lines.append(f"- **Average CVaR Max Exceedance:** {avg_cvar_max:.6f}")
            if not np.isnan(avg_var_max):
                report_lines.append(f"- **Average VaR Max Exceedance:** {avg_var_max:.6f}")
            report_lines.append("")
    
    # Tail Risk Analysis
    if "tail_risk_analysis" in report_sections:
        report_lines.append("## Tail Risk Analysis")
        report_lines.append("")
        
        if 'mean_exceedance' in metrics_df.columns:
            avg_mean_exceedance = metrics_df['mean_exceedance'].mean()
            report_lines.append(f"- **Average Mean Exceedance (VaR):** {avg_mean_exceedance:.6f}")
            report_lines.append("")
        
        if 'max_exceedance' in metrics_df.columns:
            avg_max_exceedance = metrics_df['max_exceedance'].mean()
            report_lines.append(f"- **Average Max Exceedance (VaR):** {avg_max_exceedance:.6f}")
            report_lines.append("")
        
        if 'cvar_mean_exceedance' in metrics_df.columns:
            avg_cvar_mean_exceedance = metrics_df['cvar_mean_exceedance'].mean()
            report_lines.append(f"- **Average CVaR Mean Exceedance:** {avg_cvar_mean_exceedance:.6f}")
            report_lines.append("")
        
        if 'cvar_max_exceedance' in metrics_df.columns:
            avg_cvar_max_exceedance = metrics_df['cvar_max_exceedance'].mean()
            report_lines.append(f"- **Average CVaR Max Exceedance:** {avg_cvar_max_exceedance:.6f}")
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
        
        if 'simulation_time_ms' in metrics_df.columns:
            avg_sim_time = metrics_df['simulation_time_ms'].mean()
            report_lines.append(f"- **Average Simulation Time per Portfolio:** {avg_sim_time:.2f} ms")
            report_lines.append("")
        
        if 'runtime_per_portfolio_ms' in metrics_df.columns:
            avg_runtime = metrics_df['runtime_per_portfolio_ms'].mean()
            report_lines.append(f"- **Average Total Runtime per Portfolio:** {avg_runtime:.2f} ms")
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
        
        if 'cvar_mean_exceedance' in metrics_df.columns and 'mean_exceedance' in metrics_df.columns:
            avg_cvar = metrics_df['cvar_mean_exceedance'].mean()
            avg_var = metrics_df['mean_exceedance'].mean()
            if avg_cvar > avg_var * 1.5:
                report_lines.append("- **Significant Tail Risk:** CVaR exceedances are substantially higher than VaR, indicating severe tail risk")
            report_lines.append("")
        
        if 'kurtosis' in metrics_df.columns:
            avg_kurt = metrics_df['kurtosis'].mean()
            if avg_kurt > 3:
                report_lines.append("- **Fat Tails Detected:** Returns exhibit fat tails, which Monte Carlo simulation can better capture than parametric methods")
            report_lines.append("")
        
        report_lines.append("### Recommendations")
        report_lines.append("")
        report_lines.append("- Use CVaR for portfolios with significant tail risk (when CVaR >> VaR)")
        report_lines.append("- Consider increasing number of simulations for more stable estimates")
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

