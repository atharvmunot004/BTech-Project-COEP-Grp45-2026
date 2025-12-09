"""
Report generation module for GARCH(1,1) VaR/CVaR evaluation results.

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
    garch_settings: Optional[Dict] = None,
    report_sections: Optional[List[str]] = None
) -> str:
    """
    Generate comprehensive markdown report from metrics DataFrame.
    
    Args:
        metrics_df: DataFrame with all computed metrics
        output_path: Path to save the report
        garch_settings: Dictionary with GARCH settings
        report_sections: List of sections to include
        
    Returns:
        Report content as string
    """
    if report_sections is None:
        report_sections = [
            "methodology_overview",
            "garch_model_description",
            "volatility_forecast_method",
            "var_cvar_derivation_from_garch",
            "backtesting_results",
            "tail_risk_analysis",
            "portfolio_structure_effects",
            "robustness_and_normality_checks",
            "computational_performance",
            "key_insights"
        ]
    
    report_lines = []
    
    # Header
    report_lines.append("# GARCH(1,1) Volatility Forecasting for VaR/CVaR Evaluation Report")
    report_lines.append("")
    report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Methodology Overview
    if "methodology_overview" in report_sections:
        report_lines.append("## Methodology Overview")
        report_lines.append("")
        report_lines.append("### GARCH(1,1) Volatility Forecasting for VaR and CVaR")
        report_lines.append("")
        report_lines.append("GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models capture")
        report_lines.append("time-varying volatility, which is essential for accurate risk assessment.")
        report_lines.append("VaR and CVaR are derived from GARCH volatility forecasts using distributional assumptions.")
        report_lines.append("")
        report_lines.append("**Advantages:**")
        report_lines.append("- Captures volatility clustering (high volatility followed by high volatility)")
        report_lines.append("- Adapts to changing market conditions")
        report_lines.append("- Provides dynamic risk estimates")
        report_lines.append("- Well-established in financial risk management")
        report_lines.append("")
    
    # GARCH Model Description
    if "garch_model_description" in report_sections:
        report_lines.append("## GARCH Model Description")
        report_lines.append("")
        report_lines.append("### GARCH(1,1) Model")
        report_lines.append("")
        report_lines.append("The GARCH(1,1) model specifies the conditional variance as:")
        report_lines.append("")
        report_lines.append("σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁")
        report_lines.append("")
        report_lines.append("Where:")
        report_lines.append("- σ²ₜ is the conditional variance at time t")
        report_lines.append("- ω is the constant term")
        report_lines.append("- α captures the ARCH effect (lagged squared residuals)")
        report_lines.append("- β captures the GARCH effect (lagged conditional variance)")
        report_lines.append("- εₜ are the residuals")
        report_lines.append("")
        
        if garch_settings:
            report_lines.append("### Model Parameters")
            report_lines.append("")
            report_lines.append(f"- **GARCH Order (p):** {garch_settings.get('p', 1)}")
            report_lines.append(f"- **ARCH Order (q):** {garch_settings.get('q', 1)}")
            report_lines.append(f"- **Distribution:** {garch_settings.get('distribution', 'normal')}")
            report_lines.append(f"- **Mean Model:** {'Zero' if not garch_settings.get('use_mean_model', False) else 'Constant/AR'}")
            report_lines.append(f"- **Return Type:** {garch_settings.get('return_type', 'log')}")
            report_lines.append("")
    
    # Volatility Forecast Method
    if "volatility_forecast_method" in report_sections:
        report_lines.append("## Volatility Forecast Method")
        report_lines.append("")
        report_lines.append("### Rolling Window GARCH Forecasting")
        report_lines.append("")
        report_lines.append("For each date in the evaluation period:")
        report_lines.append("1. Fit GARCH(1,1) model to a rolling window of historical returns")
        report_lines.append("2. Forecast conditional volatility for the next horizon periods")
        report_lines.append("3. Use forecasted volatility to compute VaR and CVaR")
        report_lines.append("")
        
        if garch_settings:
            report_lines.append("### Forecast Settings")
            report_lines.append("")
            report_lines.append(f"- **Forecast Method:** {garch_settings.get('forecast_method', 'rolling')}")
            report_lines.append(f"- **Estimation Windows:** {garch_settings.get('estimation_windows', [])} days")
            report_lines.append(f"- **Forecast Horizons:** {garch_settings.get('horizons', [])} days")
            report_lines.append(f"- **Fallback to Long-Run Variance:** {garch_settings.get('fallback_long_run_variance', True)}")
            report_lines.append("")
    
    # VaR/CVaR Derivation from GARCH
    if "var_cvar_derivation_from_garch" in report_sections:
        report_lines.append("## VaR/CVaR Derivation from GARCH")
        report_lines.append("")
        report_lines.append("### From Volatility to Risk Measures")
        report_lines.append("")
        report_lines.append("**VaR Calculation:**")
        report_lines.append("- VaR = σₜ₊ₕ · z_α · √h")
        report_lines.append("- Where σₜ₊ₕ is the forecasted volatility, z_α is the quantile from the assumed distribution, and h is the horizon")
        report_lines.append("")
        report_lines.append("**CVaR Calculation:**")
        report_lines.append("- CVaR = σₜ₊ₕ · ES_α · √h")
        report_lines.append("- Where ES_α is the Expected Shortfall factor for the assumed distribution")
        report_lines.append("")
        report_lines.append("**Distribution Assumptions:**")
        dist = garch_settings.get('distribution', 'normal') if garch_settings else 'normal'
        if dist == 'normal':
            report_lines.append("- Normal distribution: VaR and CVaR computed using standard normal quantiles")
        elif dist == 't':
            report_lines.append("- t-distribution: VaR and CVaR computed using Student-t quantiles (better for fat tails)")
        report_lines.append("")
        
        if garch_settings:
            report_lines.append(f"- **Confidence Levels:** {garch_settings.get('confidence_levels', [])}")
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
            report_lines.append("")
    
    # Robustness and Normality Checks
    if "robustness_and_normality_checks" in report_sections:
        report_lines.append("## Robustness and Normality Checks")
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
        
        if 'garch_fitting_time_ms' in metrics_df.columns:
            avg_fitting_time = metrics_df['garch_fitting_time_ms'].mean()
            report_lines.append(f"- **Average GARCH Fitting Time per Configuration:** {avg_fitting_time:.2f} ms")
            report_lines.append("")
        
        if 'forecast_time_ms' in metrics_df.columns:
            avg_forecast_time = metrics_df['forecast_time_ms'].mean()
            report_lines.append(f"- **Average Forecast Time per Configuration:** {avg_forecast_time:.2f} ms")
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
                report_lines.append("- **Risk Overestimation:** GARCH-based VaR tends to overestimate risk")
            elif avg_vr < 0.8:
                report_lines.append("- **Risk Underestimation:** GARCH-based VaR tends to underestimate risk")
            else:
                report_lines.append("- **Adequate Risk Estimation:** GARCH provides reasonable risk estimates")
            report_lines.append("")
        
        report_lines.append("### Recommendations")
        report_lines.append("")
        report_lines.append("- GARCH models are well-suited for portfolios with volatility clustering")
        report_lines.append("- Consider t-distribution for portfolios with fat-tailed returns")
        report_lines.append("- Monitor portfolios in 'red' traffic light zone more closely")
        report_lines.append("- Adjust estimation windows based on market regime")
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

