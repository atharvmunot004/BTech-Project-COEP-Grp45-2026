"""
Visualization tools for research results.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    print("Warning: seaborn not available, using matplotlib defaults")
from pathlib import Path
import json


def plot_risk_comparison(results_file: str, output_dir: str = None):
    """
    Plot comparison of quantum vs classical risk assessment results.
    
    Args:
        results_file: Path to experiment results JSON
        output_dir: Output directory for plots
    """
    with open(results_file, "r") as f:
        results = json.load(f)
    
    comparison_df = pd.DataFrame(results["comparison"])
    
    # Set style
    try:
        sns.set_style("whitegrid")
    except:
        pass
    plt.rcParams["figure.figsize"] = (12, 6)
    
    # Plot 1: Error comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Error vs method
    classical = comparison_df[comparison_df["type"] == "classical"]
    quantum = comparison_df[comparison_df["type"] == "quantum"]
    
    axes[0].bar(classical["method"], classical["relative_error"], 
                label="Classical", alpha=0.7, color="blue")
    axes[0].bar(quantum["method"], quantum["relative_error"], 
                label="Quantum", alpha=0.7, color="red")
    axes[0].set_xlabel("Method")
    axes[0].set_ylabel("Relative Error (%)")
    axes[0].set_title("Error Comparison: Quantum vs Classical")
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=45)
    
    # Runtime comparison
    axes[1].bar(classical["method"], classical["runtime"], 
                label="Classical", alpha=0.7, color="blue")
    axes[1].bar(quantum["method"], quantum["runtime"], 
                label="Quantum", alpha=0.7, color="red")
    axes[1].set_xlabel("Method")
    axes[1].set_ylabel("Runtime (seconds)")
    axes[1].set_title("Runtime Comparison: Quantum vs Classical")
    axes[1].legend()
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_yscale("log")
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / "risk_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    
    plt.show()


def plot_optimization_comparison(results_file: str, output_dir: str = None):
    """
    Plot comparison of quantum vs classical portfolio optimization results.
    
    Args:
        results_file: Path to experiment results JSON
        output_dir: Output directory for plots
    """
    with open(results_file, "r") as f:
        results = json.load(f)
    
    comparison_df = pd.DataFrame(results["comparison"])
    
    # Set style
    try:
        sns.set_style("whitegrid")
    except:
        pass
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    classical = comparison_df[comparison_df["type"] == "classical"]
    quantum = comparison_df[comparison_df["type"] == "quantum"]
    
    # Sharpe ratio comparison
    axes[0].bar(classical["method"], classical["sharpe_ratio"], 
                label="Classical", alpha=0.7, color="blue")
    axes[0].bar(quantum["method"], quantum["sharpe_ratio"], 
                label="Quantum", alpha=0.7, color="red")
    axes[0].set_xlabel("Method")
    axes[0].set_ylabel("Sharpe Ratio")
    axes[0].set_title("Sharpe Ratio Comparison")
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=45)
    
    # Expected return comparison
    axes[1].bar(classical["method"], classical["expected_return"], 
                label="Classical", alpha=0.7, color="blue")
    axes[1].bar(quantum["method"], quantum["expected_return"], 
                label="Quantum", alpha=0.7, color="red")
    axes[1].set_xlabel("Method")
    axes[1].set_ylabel("Expected Return")
    axes[1].set_title("Expected Return Comparison")
    axes[1].legend()
    axes[1].tick_params(axis='x', rotation=45)
    
    # Runtime comparison
    axes[2].bar(classical["method"], classical["runtime"], 
                label="Classical", alpha=0.7, color="blue")
    axes[2].bar(quantum["method"], quantum["runtime"], 
                label="Quantum", alpha=0.7, color="red")
    axes[2].set_xlabel("Method")
    axes[2].set_ylabel("Runtime (seconds)")
    axes[2].set_title("Runtime Comparison")
    axes[2].legend()
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].set_yscale("log")
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / "optimization_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    
    plt.show()


def plot_scalability(results_file: str, output_dir: str = None):
    """
    Plot scalability results.
    
    Args:
        results_file: Path to experiment results JSON
        output_dir: Output directory for plots
    """
    with open(results_file, "r") as f:
        results = json.load(f)
    
    if "scalability" not in results:
        print("No scalability data found in results")
        return
    
    scalability_data = []
    for key, value in results["scalability"].items():
        scalability_data.append(value)
    
    df = pd.DataFrame(scalability_data)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Runtime vs number of assets
    axes[0].plot(df["n_assets"], df["classical_runtime"], 
                 marker="o", label="Classical", linewidth=2)
    axes[0].plot(df["n_assets"], df["quantum_runtime"], 
                 marker="s", label="Quantum", linewidth=2)
    axes[0].set_xlabel("Number of Assets")
    axes[0].set_ylabel("Runtime (seconds)")
    axes[0].set_title("Scalability: Runtime vs Number of Assets")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Sharpe ratio vs number of assets
    axes[1].plot(df["n_assets"], df["classical_sharpe"], 
                 marker="o", label="Classical", linewidth=2)
    axes[1].plot(df["n_assets"], df["quantum_sharpe"], 
                 marker="s", label="Quantum", linewidth=2)
    axes[1].set_xlabel("Number of Assets")
    axes[1].set_ylabel("Sharpe Ratio")
    axes[1].set_title("Solution Quality: Sharpe Ratio vs Number of Assets")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / "scalability.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    
    plt.show()


if __name__ == "__main__":
    # Example usage
    results_dir = Path(__file__).parent
    
    # Plot risk comparison
    risk_results = results_dir / "experiment_01_results.json"
    if risk_results.exists():
        plot_risk_comparison(str(risk_results), output_dir=str(results_dir))
    
    # Plot optimization comparison
    opt_results = results_dir / "experiment_02_results.json"
    if opt_results.exists():
        plot_optimization_comparison(str(opt_results), output_dir=str(results_dir))
        plot_scalability(str(opt_results), output_dir=str(results_dir))

