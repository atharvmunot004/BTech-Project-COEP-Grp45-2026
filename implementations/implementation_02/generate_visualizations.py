"""
Generate comprehensive visualizations for comparative analysis
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import seaborn as sns
    HAS_SEABORN = True
    sns.set_palette("husl")
except ImportError:
    HAS_SEABORN = False

# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')

# Create output directory
output_dir = Path("results/figures")
output_dir.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load all analysis data."""
    data_dir = Path("results/comparative_analysis")
    
    risk_df = pd.read_csv(data_dir / "risk_comparison.csv")
    portfolio_df = pd.read_csv(data_dir / "portfolio_optimization_comparison.csv")
    scalability_df = pd.read_csv(data_dir / "scalability_analysis.csv")
    backtest_df = pd.read_csv(Path("results") / "backtest_results.csv")
    
    return risk_df, portfolio_df, scalability_df, backtest_df


def plot_error_vs_sample_size(risk_df):
    """
    Objective 1: Error vs Sample Size - O(1/N) vs O(1/√N) scaling
    """
    print("Generating: Error vs Sample Size Analysis...")
    
    # Filter Monte Carlo data
    mc_data = risk_df[risk_df['method'] == 'monte_carlo'].copy()
    mc_data = mc_data[mc_data['sample_size'].notna()]
    
    # Calculate error (use CVaR as reference, compare across sample sizes)
    # For each portfolio, use the largest sample size as "ground truth"
    errors = []
    
    for portfolio_id in mc_data['portfolio_id'].unique():
        portfolio_data = mc_data[mc_data['portfolio_id'] == portfolio_id]
        if len(portfolio_data) == 0:
            continue
        
        # Use 100K as ground truth
        ground_truth = portfolio_data[portfolio_data['sample_size'] == 100000.0]
        if len(ground_truth) == 0:
            # Use largest available
            ground_truth = portfolio_data.loc[portfolio_data['sample_size'].idxmax()]
        else:
            ground_truth = ground_truth.iloc[0]
        
        true_cvar = ground_truth['cvar']
        if pd.isna(true_cvar):
            continue
        
        for _, row in portfolio_data.iterrows():
            if pd.isna(row['cvar']) or row['sample_size'] == 100000.0:
                continue
            error = abs(row['cvar'] - true_cvar)
            errors.append({
                'sample_size': row['sample_size'],
                'error': error,
                'confidence_level': row['confidence_level']
            })
    
    if len(errors) == 0:
        print("  Warning: Insufficient data for error analysis")
        return
    
    error_df = pd.DataFrame(errors)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, conf_level in enumerate([0.95, 0.99]):
        ax = axes[idx]
        conf_data = error_df[error_df['confidence_level'] == conf_level]
        
        # Group by sample size and calculate mean error
        grouped = conf_data.groupby('sample_size')['error'].agg(['mean', 'std', 'count'])
        grouped = grouped.reset_index()
        
        # Plot actual errors
        ax.errorbar(grouped['sample_size'], grouped['mean'], 
                   yerr=grouped['std'], fmt='o-', label='Monte Carlo', linewidth=2, markersize=8)
        
        # Plot theoretical O(1/√N) scaling
        x_theory = np.array([1000, 5000, 10000, 50000, 100000])
        # Normalize to match first point
        y_theory = grouped['mean'].iloc[0] * np.sqrt(grouped['sample_size'].iloc[0] / x_theory)
        ax.plot(x_theory, y_theory, '--', label='O(1/√N) scaling', linewidth=2, alpha=0.7)
        
        # Plot theoretical O(1/N) scaling (for comparison)
        y_theory_n = grouped['mean'].iloc[0] * (grouped['sample_size'].iloc[0] / x_theory)
        ax.plot(x_theory, y_theory_n, '--', label='O(1/N) scaling', linewidth=2, alpha=0.7)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Sample Size (N)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Absolute Error', fontsize=12, fontweight='bold')
        ax.set_title(f'Error vs Sample Size (Confidence Level: {int(conf_level*100)}%)', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "error_vs_sample_size.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'error_vs_sample_size.png'}")


def plot_runtime_comparison(risk_df):
    """Runtime comparison across risk methods."""
    print("Generating: Runtime Comparison...")
    
    # Calculate average runtime by method
    runtime_stats = risk_df.groupby('method')['runtime'].agg(['mean', 'std', 'count']).reset_index()
    runtime_stats = runtime_stats.sort_values('mean')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.barh(runtime_stats['method'], runtime_stats['mean'], 
                   xerr=runtime_stats['std'], capsize=5, alpha=0.8)
    
    # Color bars
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, bar in enumerate(bars):
        bar.set_color(colors[i % len(colors)])
    
    ax.set_xlabel('Average Runtime (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Method', fontsize=12, fontweight='bold')
    ax.set_title('Runtime Comparison: Risk Assessment Methods', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (method, mean, std) in enumerate(zip(runtime_stats['method'], 
                                                  runtime_stats['mean'], 
                                                  runtime_stats['std'])):
        ax.text(mean + std + 0.0001, i, f'{mean:.4f}s', 
               va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "runtime_comparison_risk.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'runtime_comparison_risk.png'}")


def plot_var_cvar_comparison(risk_df):
    """VaR/CVaR estimates comparison across methods."""
    print("Generating: VaR/CVaR Comparison...")
    
    # Filter data with valid estimates
    valid_data = risk_df[risk_df['cvar'].notna() | risk_df['var'].notna()].copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # CVaR comparison
    ax = axes[0]
    cvar_data = valid_data[valid_data['cvar'].notna()]
    if len(cvar_data) > 0:
        methods = cvar_data['method'].unique()
        positions = np.arange(len(methods))
        means = [cvar_data[cvar_data['method'] == m]['cvar'].mean() for m in methods]
        stds = [cvar_data[cvar_data['method'] == m]['cvar'].std() for m in methods]
        
        bars = ax.bar(positions, means, yerr=stds, capsize=5, alpha=0.8, 
                     color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(methods)])
        ax.set_xticks(positions)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_ylabel('CVaR Estimate', fontsize=12, fontweight='bold')
        ax.set_title('CVaR Estimates by Method', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    # VaR comparison
    ax = axes[1]
    var_data = valid_data[valid_data['var'].notna()]
    if len(var_data) > 0:
        methods = var_data['method'].unique()
        positions = np.arange(len(methods))
        means = [var_data[var_data['method'] == m]['var'].mean() for m in methods]
        stds = [var_data[var_data['method'] == m]['var'].std() for m in methods]
        
        bars = ax.bar(positions, means, yerr=stds, capsize=5, alpha=0.8,
                     color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(methods)])
        ax.set_xticks(positions)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_ylabel('VaR Estimate', fontsize=12, fontweight='bold')
        ax.set_title('VaR Estimates by Method', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / "var_cvar_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'var_cvar_comparison.png'}")


def plot_portfolio_performance(portfolio_df, backtest_df):
    """Portfolio optimization performance comparison."""
    print("Generating: Portfolio Performance Comparison...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Sharpe Ratio Comparison
    ax = axes[0, 0]
    methods = portfolio_df['method'].unique()
    sharpe_values = []
    method_labels = []
    
    for method in methods:
        method_data = portfolio_df[portfolio_df['method'] == method]
        if 'avg_sharpe' in method_data.columns and method_data['avg_sharpe'].notna().any():
            sharpe = method_data['avg_sharpe'].iloc[0]
            sharpe_values.append(sharpe)
            method_labels.append(method)
    
    if sharpe_values:
        bars = ax.bar(range(len(method_labels)), sharpe_values, alpha=0.8,
                     color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(method_labels)])
        ax.set_xticks(range(len(method_labels)))
        ax.set_xticklabels(method_labels, rotation=45, ha='right')
        ax.set_ylabel('Average Sharpe Ratio', fontsize=12, fontweight='bold')
        ax.set_title('Backtested Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, v in enumerate(sharpe_values):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    
    # 2. Return vs Volatility (Risk-Return Tradeoff)
    ax = axes[0, 1]
    for method in methods:
        method_data = portfolio_df[portfolio_df['method'] == method]
        if len(method_data) > 0:
            row = method_data.iloc[0]
            if pd.notna(row.get('avg_return')) and pd.notna(row.get('avg_volatility')):
                ax.scatter(row['avg_volatility'], row['avg_return'], 
                          s=200, alpha=0.7, label=method)
    
    ax.set_xlabel('Volatility (Annualized)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Return (Annualized)', fontsize=12, fontweight='bold')
    ax.set_title('Risk-Return Tradeoff', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Runtime Comparison
    ax = axes[1, 0]
    runtime_data = portfolio_df[['method', 'runtime']].dropna()
    if len(runtime_data) > 0:
        runtime_stats = runtime_data.groupby('method')['runtime'].mean().sort_values()
        bars = ax.barh(runtime_stats.index, runtime_stats.values, alpha=0.8,
                      color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(runtime_stats)])
        ax.set_xlabel('Average Runtime (seconds)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Method', fontsize=12, fontweight='bold')
        ax.set_title('Portfolio Optimization Runtime', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    # 4. Solution Sparsity
    ax = axes[1, 1]
    sparsity_data = portfolio_df[['method', 'sparsity']].dropna()
    if len(sparsity_data) > 0:
        methods = sparsity_data['method'].unique()
        sparsity_values = [sparsity_data[sparsity_data['method'] == m]['sparsity'].iloc[0] 
                          for m in methods]
        bars = ax.bar(range(len(methods)), sparsity_values, alpha=0.8,
                     color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(methods)])
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_ylabel('Sparsity (Fraction of Non-Zero Weights)', fontsize=12, fontweight='bold')
        ax.set_title('Solution Sparsity Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(output_dir / "portfolio_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'portfolio_performance.png'}")


def plot_backtest_timeseries(backtest_df):
    """Backtest performance over time."""
    print("Generating: Backtest Time Series...")
    
    if len(backtest_df) == 0:
        print("  Warning: No backtest data available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Group by method
    methods = backtest_df['method'].unique()
    
    # 1. Sharpe Ratio over Windows
    ax = axes[0, 0]
    for method in methods:
        method_data = backtest_df[backtest_df['method'] == method].sort_values('window_id')
        ax.plot(method_data['window_id'], method_data['sharpe_ratio'], 
               marker='o', label=method, linewidth=2, markersize=6)
    ax.set_xlabel('Window ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
    ax.set_title('Sharpe Ratio Across Rolling Windows', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Return over Windows
    ax = axes[0, 1]
    for method in methods:
        method_data = backtest_df[backtest_df['method'] == method].sort_values('window_id')
        ax.plot(method_data['window_id'], method_data['annualized_return'], 
               marker='s', label=method, linewidth=2, markersize=6)
    ax.set_xlabel('Window ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Annualized Return', fontsize=12, fontweight='bold')
    ax.set_title('Returns Across Rolling Windows', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Volatility over Windows
    ax = axes[1, 0]
    for method in methods:
        method_data = backtest_df[backtest_df['method'] == method].sort_values('window_id')
        ax.plot(method_data['window_id'], method_data['annualized_volatility'], 
               marker='^', label=method, linewidth=2, markersize=6)
    ax.set_xlabel('Window ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Annualized Volatility', fontsize=12, fontweight='bold')
    ax.set_title('Volatility Across Rolling Windows', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Max Drawdown over Windows
    ax = axes[1, 1]
    for method in methods:
        method_data = backtest_df[backtest_df['method'] == method].sort_values('window_id')
        ax.plot(method_data['window_id'], method_data['max_drawdown'], 
               marker='d', label=method, linewidth=2, markersize=6)
    ax.set_xlabel('Window ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Max Drawdown', fontsize=12, fontweight='bold')
    ax.set_title('Max Drawdown Across Rolling Windows', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "backtest_timeseries.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'backtest_timeseries.png'}")


def plot_scalability_analysis(scalability_df):
    """Scalability analysis visualization."""
    print("Generating: Scalability Analysis...")
    
    if len(scalability_df) == 0:
        print("  Warning: No scalability data available")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    methods = scalability_df['method'].unique()
    
    # 1. Runtime vs Number of Assets
    ax = axes[0]
    for method in methods:
        method_data = scalability_df[scalability_df['method'] == method].sort_values('num_assets')
        ax.plot(method_data['num_assets'], method_data['runtime'], 
               marker='o', label=method, linewidth=2, markersize=8)
    ax.set_xlabel('Number of Assets', fontsize=12, fontweight='bold')
    ax.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Runtime Scaling with Number of Assets', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Sharpe Ratio vs Number of Assets
    ax = axes[1]
    for method in methods:
        method_data = scalability_df[scalability_df['method'] == method].sort_values('num_assets')
        ax.plot(method_data['num_assets'], method_data['sharpe_ratio'], 
               marker='s', label=method, linewidth=2, markersize=8)
    ax.set_xlabel('Number of Assets', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
    ax.set_title('Performance vs Number of Assets', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "scalability_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'scalability_analysis.png'}")


def plot_method_comparison_summary(risk_df, portfolio_df):
    """Comprehensive method comparison summary."""
    print("Generating: Method Comparison Summary...")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Risk Methods - Average CVaR
    ax1 = fig.add_subplot(gs[0, 0])
    cvar_data = risk_df[risk_df['cvar'].notna()]
    if len(cvar_data) > 0:
        cvar_means = cvar_data.groupby('method')['cvar'].mean()
        bars = ax1.bar(range(len(cvar_means)), cvar_means.values, alpha=0.8,
                     color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(cvar_means)])
        ax1.set_xticks(range(len(cvar_means)))
        ax1.set_xticklabels(cvar_means.index, rotation=45, ha='right', fontsize=9)
        ax1.set_ylabel('Avg CVaR', fontsize=10, fontweight='bold')
        ax1.set_title('Risk: CVaR Estimates', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Risk Methods - Runtime
    ax2 = fig.add_subplot(gs[0, 1])
    runtime_means = risk_df.groupby('method')['runtime'].mean().sort_values()
    bars = ax2.barh(range(len(runtime_means)), runtime_means.values, alpha=0.8,
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(runtime_means)])
    ax2.set_yticks(range(len(runtime_means)))
    ax2.set_yticklabels(runtime_means.index, fontsize=9)
    ax2.set_xlabel('Runtime (s)', fontsize=10, fontweight='bold')
    ax2.set_title('Risk: Runtime', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Portfolio Methods - Sharpe Ratio
    ax3 = fig.add_subplot(gs[0, 2])
    if 'avg_sharpe' in portfolio_df.columns:
        sharpe_data = portfolio_df[['method', 'avg_sharpe']].dropna()
        if len(sharpe_data) > 0:
            bars = ax3.bar(range(len(sharpe_data)), sharpe_data['avg_sharpe'].values, 
                          alpha=0.8, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(sharpe_data)])
            ax3.set_xticks(range(len(sharpe_data)))
            ax3.set_xticklabels(sharpe_data['method'], rotation=45, ha='right', fontsize=9)
            ax3.set_ylabel('Avg Sharpe', fontsize=10, fontweight='bold')
            ax3.set_title('Portfolio: Sharpe Ratio', fontsize=11, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Portfolio Methods - Return
    ax4 = fig.add_subplot(gs[1, 0])
    if 'avg_return' in portfolio_df.columns:
        return_data = portfolio_df[['method', 'avg_return']].dropna()
        if len(return_data) > 0:
            bars = ax4.bar(range(len(return_data)), return_data['avg_return'].values, 
                          alpha=0.8, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(return_data)])
            ax4.set_xticks(range(len(return_data)))
            ax4.set_xticklabels(return_data['method'], rotation=45, ha='right', fontsize=9)
            ax4.set_ylabel('Avg Return', fontsize=10, fontweight='bold')
            ax4.set_title('Portfolio: Returns', fontsize=11, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Portfolio Methods - Volatility
    ax5 = fig.add_subplot(gs[1, 1])
    if 'avg_volatility' in portfolio_df.columns:
        vol_data = portfolio_df[['method', 'avg_volatility']].dropna()
        if len(vol_data) > 0:
            bars = ax5.bar(range(len(vol_data)), vol_data['avg_volatility'].values, 
                          alpha=0.8, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(vol_data)])
            ax5.set_xticks(range(len(vol_data)))
            ax5.set_xticklabels(vol_data['method'], rotation=45, ha='right', fontsize=9)
            ax5.set_ylabel('Avg Volatility', fontsize=10, fontweight='bold')
            ax5.set_title('Portfolio: Volatility', fontsize=11, fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Portfolio Methods - Max Drawdown
    ax6 = fig.add_subplot(gs[1, 2])
    if 'avg_max_drawdown' in portfolio_df.columns:
        dd_data = portfolio_df[['method', 'avg_max_drawdown']].dropna()
        if len(dd_data) > 0:
            bars = ax6.bar(range(len(dd_data)), dd_data['avg_max_drawdown'].values, 
                          alpha=0.8, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(dd_data)])
            ax6.set_xticks(range(len(dd_data)))
            ax6.set_xticklabels(dd_data['method'], rotation=45, ha='right', fontsize=9)
            ax6.set_ylabel('Avg Max DD', fontsize=10, fontweight='bold')
            ax6.set_title('Portfolio: Max Drawdown', fontsize=11, fontweight='bold')
            ax6.grid(True, alpha=0.3, axis='y')
    
    # 7-9. Sample size analysis (if available)
    ax7 = fig.add_subplot(gs[2, :])
    mc_data = risk_df[risk_df['method'] == 'monte_carlo']
    if len(mc_data) > 0 and mc_data['sample_size'].notna().any():
        sample_sizes = sorted(mc_data['sample_size'].dropna().unique())
        avg_runtime = [mc_data[mc_data['sample_size'] == s]['runtime'].mean() 
                      for s in sample_sizes]
        ax7.plot(sample_sizes, avg_runtime, 'o-', linewidth=2, markersize=8, label='Monte Carlo')
        ax7.set_xscale('log')
        ax7.set_xlabel('Sample Size', fontsize=12, fontweight='bold')
        ax7.set_ylabel('Average Runtime (s)', fontsize=12, fontweight='bold')
        ax7.set_title('Monte Carlo: Runtime vs Sample Size', fontsize=14, fontweight='bold')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
    
    plt.suptitle('Comprehensive Method Comparison Summary', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(output_dir / "method_comparison_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'method_comparison_summary.png'}")


def main():
    """Generate all visualizations."""
    print("="*80)
    print("GENERATING COMPREHENSIVE VISUALIZATIONS")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    risk_df, portfolio_df, scalability_df, backtest_df = load_data()
    
    print(f"  Risk data: {len(risk_df)} rows")
    print(f"  Portfolio data: {len(portfolio_df)} rows")
    print(f"  Scalability data: {len(scalability_df)} rows")
    print(f"  Backtest data: {len(backtest_df)} rows")
    
    # Generate all plots
    print("\nGenerating visualizations...")
    
    plot_error_vs_sample_size(risk_df)
    plot_runtime_comparison(risk_df)
    plot_var_cvar_comparison(risk_df)
    plot_portfolio_performance(portfolio_df, backtest_df)
    plot_backtest_timeseries(backtest_df)
    plot_scalability_analysis(scalability_df)
    plot_method_comparison_summary(risk_df, portfolio_df)
    
    print("\n" + "="*80)
    print("VISUALIZATION GENERATION COMPLETE!")
    print("="*80)
    print(f"\nAll figures saved to: {output_dir}")
    print("\nGenerated figures:")
    for fig_file in sorted(output_dir.glob("*.png")):
        print(f"  - {fig_file.name}")


if __name__ == "__main__":
    main()

