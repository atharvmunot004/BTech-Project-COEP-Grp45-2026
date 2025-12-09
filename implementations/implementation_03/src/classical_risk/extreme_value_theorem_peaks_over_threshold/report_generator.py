"""
Report generation module for EVT-POT VaR/CVaR evaluation results.

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
    evt_settings: Optional[Dict] = None,
    report_sections: Optional[List[str]] = None
) -> str:
    """
    Generate comprehensive markdown report from metrics DataFrame.
    
    Args:
        metrics_df: DataFrame with all computed metrics
        output_path: Path to save the report
        evt_settings: Dictionary with EVT settings
        report_sections: List of sections to include
        
    Returns:
        Report content as string
    """
    if report_sections is None:
        report_sections = [
            "methodology_overview",
            "evt_theory_and_pot_framework",
            "threshold_selection_procedure",
            "gpd_parameter_estimation",
            "evt_based_var_cvar_calculation",
            "backtesting_results",
            "tail_risk_behavior",
            "portfolio_structure_effects",
            "stability_and_robustness_checks",
            "computational_performance",
            "key_insights"
        ]
    
    report_lines = []
    
    # Header
    report_lines.append("# Extreme Value Theory (EVT) - Peaks Over Threshold (POT) VaR/CVaR Evaluation Report")
    report_lines.append("")
    report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Methodology Overview
    if "methodology_overview" in report_sections:
        report_lines.append("## Methodology Overview")
        report_lines.append("")
        report_lines.append("### Extreme Value Theory - Peaks Over Threshold")
        report_lines.append("")
        report_lines.append("EVT-POT is a statistical approach for modeling extreme tail events in financial returns.")
        report_lines.append("It focuses on exceedances over a high threshold and fits a Generalized Pareto Distribution (GPD)")
        report_lines.append("to model the tail behavior, providing more accurate estimates of extreme risk measures.")
        report_lines.append("")
        report_lines.append("**Advantages:**")
        report_lines.append("- Specifically designed for extreme tail events")
        report_lines.append("- Captures heavy-tailed distributions better than parametric methods")
        report_lines.append("- Provides tail index (shape parameter) for tail risk assessment")
        report_lines.append("- More robust for high confidence levels (e.g., 99%, 99.5%)")
        report_lines.append("")
    
    # EVT Theory and POT Framework
    if "evt_theory_and_pot_framework" in report_sections:
        report_lines.append("## EVT Theory and POT Framework")
        report_lines.append("")
        report_lines.append("### Theoretical Foundation")
        report_lines.append("")
        report_lines.append("Extreme Value Theory (EVT) provides a framework for modeling the tail behavior of distributions.")
        report_lines.append("The Peaks Over Threshold (POT) method focuses on observations exceeding a high threshold.")
        report_lines.append("")
        report_lines.append("**Key Theorem:** Pickands-Balkema-de Haan Theorem")
        report_lines.append("- For a sufficiently high threshold u, the distribution of exceedances (X - u | X > u)")
        report_lines.append("  converges to a Generalized Pareto Distribution (GPD) as u increases.")
        report_lines.append("")
        report_lines.append("**GPD Distribution:**")
        report_lines.append("- Shape parameter (ξ): Determines tail behavior")
        report_lines.append("  - ξ > 0: Heavy-tailed (Pareto-type)")
        report_lines.append("  - ξ = 0: Exponential tail")
        report_lines.append("  - ξ < 0: Light-tailed (bounded)")
        report_lines.append("- Scale parameter (β): Controls dispersion")
        report_lines.append("")
    
    # Threshold Selection Procedure
    if "threshold_selection_procedure" in report_sections:
        report_lines.append("## Threshold Selection Procedure")
        report_lines.append("")
        report_lines.append("### Threshold Selection Methods")
        report_lines.append("")
        
        if evt_settings:
            threshold_settings = evt_settings.get('threshold_selection', {})
            method = threshold_settings.get('method', 'quantile')
            quantiles = threshold_settings.get('quantiles', [0.90, 0.95, 0.97])
            automatic = threshold_settings.get('automatic_threshold', True)
            min_exceedances = threshold_settings.get('min_exceedances', 50)
            
            report_lines.append(f"- **Selection Method:** {method}")
            report_lines.append(f"- **Automatic Selection:** {automatic}")
            report_lines.append(f"- **Quantiles Tested:** {quantiles}")
            report_lines.append(f"- **Minimum Exceedances Required:** {min_exceedances}")
            report_lines.append("")
            
            if automatic:
                report_lines.append("**Automatic Selection Process:**")
                report_lines.append("1. Test multiple quantile thresholds")
                report_lines.append("2. Evaluate mean excess and stability")
                report_lines.append("3. Select threshold with sufficient exceedances and stability")
                report_lines.append("")
    
    # GPD Parameter Estimation
    if "gpd_parameter_estimation" in report_sections:
        report_lines.append("## GPD Parameter Estimation")
        report_lines.append("")
        report_lines.append("### Maximum Likelihood Estimation")
        report_lines.append("")
        report_lines.append("GPD parameters are estimated using Maximum Likelihood Estimation (MLE) on exceedances.")
        report_lines.append("")
        
        if evt_settings:
            shape_constraints = evt_settings.get('shape_constraints', {})
            xi_lower = shape_constraints.get('xi_lower_bound', -0.5)
            xi_upper = shape_constraints.get('xi_upper_bound', 0.5)
            fitting_method = evt_settings.get('gpd_fitting_method', 'mle')
            
            report_lines.append(f"- **Fitting Method:** {fitting_method}")
            report_lines.append(f"- **Shape Parameter (ξ) Bounds:** [{xi_lower}, {xi_upper}]")
            report_lines.append("")
            
            if 'tail_index_xi' in metrics_df.columns:
                avg_xi = metrics_df['tail_index_xi'].mean()
                report_lines.append(f"- **Average Tail Index (ξ):** {avg_xi:.4f}")
                if avg_xi > 0:
                    report_lines.append("  - Positive ξ indicates heavy-tailed distribution (Pareto-type)")
                elif avg_xi < 0:
                    report_lines.append("  - Negative ξ indicates light-tailed distribution (bounded)")
                else:
                    report_lines.append("  - ξ ≈ 0 indicates exponential tail")
                report_lines.append("")
    
    # EVT-Based VaR/CVaR Calculation
    if "evt_based_var_cvar_calculation" in report_sections:
        report_lines.append("## EVT-Based VaR/CVaR Calculation")
        report_lines.append("")
        report_lines.append("### From GPD to Risk Measures")
        report_lines.append("")
        report_lines.append("**VaR Calculation:**")
        report_lines.append("- VaR is computed using the GPD distribution fitted to exceedances")
        report_lines.append("- Accounts for probability of exceedance and tail behavior")
        report_lines.append("- Formula: VaR = u + (β/ξ) · [((n/nu) · (1-α))^(-ξ) - 1]")
        report_lines.append("  - Where u is threshold, β is scale, ξ is shape, n is sample size, nu is exceedances, α is confidence level")
        report_lines.append("")
        report_lines.append("**CVaR Calculation:**")
        report_lines.append("- CVaR (Expected Shortfall) is the expected loss given VaR is exceeded")
        report_lines.append("- Formula: CVaR = VaR + (β - ξ·(VaR - u)) / (1 - ξ) for ξ < 1")
        report_lines.append("")
        
        if evt_settings:
            report_lines.append(f"- **Confidence Levels:** {evt_settings.get('confidence_levels', [])}")
            report_lines.append(f"- **Horizons:** {evt_settings.get('horizons', [])} days")
            report_lines.append(f"- **Estimation Windows:** {evt_settings.get('estimation_windows', [])} days")
            report_lines.append("")
    
    # Summary Statistics
    report_lines.append("## Summary Statistics")
    report_lines.append("")
    report_lines.append(f"- **Total Portfolio-Configuration Combinations:** {len(metrics_df)}")
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
                report_lines.append(f"- **{zone.capitalize()}:** {count} configurations ({count/len(metrics_df)*100:.1f}%)")
            report_lines.append("")
        
        if 'kupiec_unconditional_coverage' in metrics_df.columns:
            kupiec_passed = (metrics_df['kupiec_unconditional_coverage'] > 0.05).sum()
            report_lines.append(f"- **Kupiec Test Passed:** {kupiec_passed}/{len(metrics_df)} configurations ({kupiec_passed/len(metrics_df)*100:.1f}%)")
            report_lines.append("")
    
    # Tail Risk Behavior
    if "tail_risk_behavior" in report_sections:
        report_lines.append("## Tail Risk Behavior")
        report_lines.append("")
        
        if 'mean_exceedance' in metrics_df.columns:
            avg_mean_exceedance = metrics_df['mean_exceedance'].mean()
            report_lines.append(f"- **Average Mean Exceedance (VaR):** {avg_mean_exceedance:.6f}")
            report_lines.append("")
        
        if 'max_exceedance' in metrics_df.columns:
            avg_max_exceedance = metrics_df['max_exceedance'].mean()
            report_lines.append(f"- **Average Max Exceedance (VaR):** {avg_max_exceedance:.6f}")
            report_lines.append("")
        
        if 'expected_shortfall_exceedance' in metrics_df.columns:
            avg_es_exceedance = metrics_df['expected_shortfall_exceedance'].mean()
            report_lines.append(f"- **Average Expected Shortfall Exceedance:** {avg_es_exceedance:.6f}")
            report_lines.append("")
        
        if 'cvar_mean_exceedance' in metrics_df.columns:
            avg_cvar_mean_exceedance = metrics_df['cvar_mean_exceedance'].mean()
            report_lines.append(f"- **Average CVaR Mean Exceedance:** {avg_cvar_mean_exceedance:.6f}")
            report_lines.append("")
        
        if 'cvar_max_exceedance' in metrics_df.columns:
            avg_cvar_max_exceedance = metrics_df['cvar_max_exceedance'].mean()
            report_lines.append(f"- **Average CVaR Max Exceedance:** {avg_cvar_max_exceedance:.6f}")
            report_lines.append("")
        
        if 'tail_index_xi' in metrics_df.columns:
            avg_xi = metrics_df['tail_index_xi'].mean()
            report_lines.append(f"- **Average Tail Index (ξ):** {avg_xi:.4f}")
            report_lines.append("")
        
        if 'shape_scale_stability' in metrics_df.columns:
            avg_stability = metrics_df['shape_scale_stability'].mean()
            report_lines.append(f"- **Average Shape-Scale Stability:** {avg_stability:.4f}")
            report_lines.append("  - Lower values indicate more stable tail behavior")
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
            report_lines.append("")
    
    # Stability and Robustness Checks
    if "stability_and_robustness_checks" in report_sections:
        report_lines.append("## Stability and Robustness Checks")
        report_lines.append("")
        
        if 'tail_index_xi' in metrics_df.columns:
            xi_std = metrics_df['tail_index_xi'].std()
            report_lines.append(f"- **Tail Index (ξ) Standard Deviation:** {xi_std:.4f}")
            report_lines.append("  - Lower values indicate more stable tail behavior across portfolios")
            report_lines.append("")
        
        if 'shape_scale_stability' in metrics_df.columns:
            avg_stability = metrics_df['shape_scale_stability'].mean()
            report_lines.append(f"- **Average Shape-Scale Stability:** {avg_stability:.4f}")
            report_lines.append("")
        
        if 'skewness' in metrics_df.columns:
            avg_skewness = metrics_df['skewness'].mean()
            report_lines.append(f"- **Average Skewness:** {avg_skewness:.4f}")
            report_lines.append("")
        
        if 'kurtosis' in metrics_df.columns:
            avg_kurtosis = metrics_df['kurtosis'].mean()
            report_lines.append(f"- **Average Excess Kurtosis:** {avg_kurtosis:.4f}")
            report_lines.append("")
        
        if 'jarque_bera_p_value' in metrics_df.columns:
            normality_passed = (metrics_df['jarque_bera_p_value'] > 0.05).sum()
            report_lines.append(f"- **Normality Tests Passed (Jarque-Bera):** {normality_passed}/{len(metrics_df)} configurations ({normality_passed/len(metrics_df)*100:.1f}%)")
            report_lines.append("")
    
    # Computational Performance
    if "computational_performance" in report_sections:
        report_lines.append("## Computational Performance")
        report_lines.append("")
        
        if 'evt_fitting_time_ms' in metrics_df.columns:
            avg_fitting_time = metrics_df['evt_fitting_time_ms'].mean()
            report_lines.append(f"- **Average EVT Fitting Time per Configuration:** {avg_fitting_time:.2f} ms")
            report_lines.append("")
        
        if 'threshold_selection_time_ms' in metrics_df.columns:
            avg_threshold_time = metrics_df['threshold_selection_time_ms'].mean()
            report_lines.append(f"- **Average Threshold Selection Time per Configuration:** {avg_threshold_time:.2f} ms")
            report_lines.append("")
        
        if 'runtime_per_portfolio_ms' in metrics_df.columns:
            avg_runtime = metrics_df['runtime_per_portfolio_ms'].mean()
            report_lines.append(f"- **Average Total Runtime per Portfolio:** {avg_runtime:.2f} ms")
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
                report_lines.append("- **Risk Overestimation:** EVT-POT tends to overestimate risk")
            elif avg_vr < 0.8:
                report_lines.append("- **Risk Underestimation:** EVT-POT tends to underestimate risk")
            else:
                report_lines.append("- **Adequate Risk Estimation:** EVT-POT provides reasonable risk estimates")
            report_lines.append("")
        
        if 'tail_index_xi' in metrics_df.columns:
            avg_xi = metrics_df['tail_index_xi'].mean()
            if avg_xi > 0.1:
                report_lines.append("- **Heavy-Tailed Distribution:** Positive tail index indicates heavy tails")
                report_lines.append("  - EVT-POT is well-suited for such distributions")
            elif avg_xi < -0.1:
                report_lines.append("- **Light-Tailed Distribution:** Negative tail index indicates bounded tails")
            else:
                report_lines.append("- **Moderate Tail Behavior:** Tail index near zero indicates exponential tails")
            report_lines.append("")
        
        report_lines.append("### Recommendations")
        report_lines.append("")
        report_lines.append("- EVT-POT is particularly effective for high confidence levels (99%, 99.5%)")
        report_lines.append("- Monitor tail index (ξ) for stability across portfolios")
        report_lines.append("- Adjust threshold selection based on available data and required exceedances")
        report_lines.append("- Consider EVT-POT for portfolios with heavy-tailed return distributions")
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
    
    # Write report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report_content = "\n".join(report_lines)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return report_content

