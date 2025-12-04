"""
Research Experiment 1: Quantum vs Classical Risk Assessment

Benchmark QAE against classical Monte Carlo for VaR and CVaR estimation.
Measures empirical error vs. sample size to show O(1/N) vs O(1/sqrt(N)) scaling.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import json
from datetime import datetime
import time

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from data.preprocessing_pipeline import PreprocessingPipeline
from classical.risk.monte_carlo import monte_carlo_var_cvar, historical_var_cvar
from classical.risk.parametric_var import parametric_var
from classical.risk.cvar import calculate_cvar
from quantum.qae_cvar import QAECVaREstimator


class RiskBenchmarkExperiment:
    """Benchmark quantum vs classical risk assessment methods."""
    
    def __init__(
        self,
        symbols: list,
        confidence_level: float = 0.95,
        random_seed: int = 42
    ):
        """
        Initialize experiment.
        
        Args:
            symbols: List of stock symbols
            confidence_level: Confidence level for VaR/CVaR
            random_seed: Random seed for reproducibility
        """
        self.symbols = symbols
        self.confidence_level = confidence_level
        self.random_seed = random_seed
        
        # Load and preprocess data
        print("Loading and preprocessing data...")
        self.pipeline = PreprocessingPipeline()
        self.data = self.pipeline.load_and_preprocess(symbols)
        
        # Equal weights portfolio
        self.portfolio_weights = np.ones(len(symbols)) / len(symbols)
        
        # Get returns
        self.returns = self.data["returns_df"]
    
    def run_classical_baseline(self) -> dict:
        """Run classical risk assessment methods."""
        print("\n=== Running Classical Risk Assessment ===")
        
        results = {}
        
        # Parametric VaR
        start_time = time.time()
        var_param = parametric_var(
            self.returns,
            portfolio_weights=self.portfolio_weights,
            confidence_level=self.confidence_level
        )
        results["parametric_var"] = {
            "value": float(var_param),
            "runtime": time.time() - start_time
        }
        print(f"Parametric VaR: {var_param:.4f}")
        
        # Historical VaR/CVaR
        start_time = time.time()
        var_hist, cvar_hist = historical_var_cvar(
            self.returns,
            portfolio_weights=self.portfolio_weights,
            confidence_level=self.confidence_level
        )
        results["historical_var"] = {
            "value": float(var_hist),
            "runtime": time.time() - start_time
        }
        results["historical_cvar"] = {
            "value": float(cvar_hist),
            "runtime": time.time() - start_time
        }
        print(f"Historical VaR: {var_hist:.4f}, CVaR: {cvar_hist:.4f}")
        
        # Monte Carlo VaR/CVaR with different sample sizes
        sample_sizes = [1000, 5000, 10000, 50000, 100000]
        results["monte_carlo"] = {}
        
        for n_sims in sample_sizes:
            start_time = time.time()
            var_mc, cvar_mc, _ = monte_carlo_var_cvar(
                self.returns,
                portfolio_weights=self.portfolio_weights,
                confidence_level=self.confidence_level,
                n_simulations=n_sims,
                random_seed=self.random_seed
            )
            runtime = time.time() - start_time
            
            results["monte_carlo"][f"n_{n_sims}"] = {
                "var": float(var_mc),
                "cvar": float(cvar_mc),
                "runtime": runtime,
                "n_simulations": n_sims
            }
            print(f"Monte Carlo (n={n_sims}): VaR={var_mc:.4f}, CVaR={cvar_mc:.4f}, Time={runtime:.2f}s")
        
        # Analytical CVaR
        start_time = time.time()
        cvar_analytical = calculate_cvar(
            self.returns,
            portfolio_weights=self.portfolio_weights,
            confidence_level=self.confidence_level,
            method="analytical"
        )
        results["analytical_cvar"] = {
            "value": float(cvar_analytical),
            "runtime": time.time() - start_time
        }
        print(f"Analytical CVaR: {cvar_analytical:.4f}")
        
        return results
    
    def run_quantum_risk(self, n_qubits_list: list = [3, 4, 5]) -> dict:
        """Run quantum risk assessment methods."""
        print("\n=== Running Quantum Risk Assessment ===")
        
        results = {}
        
        # QAE CVaR with different qubit counts
        for n_qubits in n_qubits_list:
            print(f"\nQAE with {n_qubits} qubits...")
            start_time = time.time()
            
            qae_estimator = QAECVaREstimator(
                n_qubits=n_qubits,
                random_seed=self.random_seed
            )
            
            qae_result = qae_estimator.estimate_cvar(
                self.returns,
                portfolio_weights=self.portfolio_weights,
                confidence_level=self.confidence_level
            )
            
            runtime = time.time() - start_time
            
            results[f"qae_n_qubits_{n_qubits}"] = {
                "cvar_estimate": float(qae_result["cvar_estimate"]),
                "classical_cvar": float(qae_result["classical_cvar"]),
                "var_threshold": float(qae_result["var_threshold"]),
                "runtime": runtime,
                "n_qubits": n_qubits,
                "n_evaluation_qubits": qae_result["n_evaluation_qubits"],
                "shots": qae_result["shots"]
            }
            
            print(f"  QAE CVaR: {qae_result['cvar_estimate']:.4f}")
            print(f"  Classical CVaR: {qae_result['classical_cvar']:.4f}")
            print(f"  Runtime: {runtime:.2f}s")
        
        return results
    
    def compare_results(self, classical_results: dict, quantum_results: dict) -> pd.DataFrame:
        """Compare classical and quantum results."""
        comparison_data = []
        
        # Classical methods
        comparison_data.append({
            "method": "Parametric VaR",
            "type": "classical",
            "value": classical_results["parametric_var"]["value"],
            "runtime": classical_results["parametric_var"]["runtime"]
        })
        
        comparison_data.append({
            "method": "Historical CVaR",
            "type": "classical",
            "value": classical_results["historical_cvar"]["value"],
            "runtime": classical_results["historical_cvar"]["runtime"]
        })
        
        comparison_data.append({
            "method": "Analytical CVaR",
            "type": "classical",
            "value": classical_results["analytical_cvar"]["value"],
            "runtime": classical_results["analytical_cvar"]["runtime"]
        })
        
        # Monte Carlo with different sample sizes
        for key, value in classical_results["monte_carlo"].items():
            comparison_data.append({
                "method": f"Monte Carlo CVaR (n={value['n_simulations']})",
                "type": "classical",
                "value": value["cvar"],
                "runtime": value["runtime"],
                "n_simulations": value["n_simulations"]
            })
        
        # Quantum methods
        for key, value in quantum_results.items():
            comparison_data.append({
                "method": f"QAE CVaR (n_qubits={value['n_qubits']})",
                "type": "quantum",
                "value": value["cvar_estimate"],
                "runtime": value["runtime"],
                "n_qubits": value["n_qubits"]
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Calculate error relative to analytical CVaR (ground truth)
        ground_truth = classical_results["analytical_cvar"]["value"]
        df["error"] = np.abs(df["value"] - ground_truth)
        df["relative_error"] = df["error"] / ground_truth * 100
        
        return df
    
    def run_full_experiment(self) -> dict:
        """Run complete experiment."""
        print("=" * 60)
        print("Research Experiment 1: Quantum vs Classical Risk Assessment")
        print("=" * 60)
        
        # Run classical methods
        classical_results = self.run_classical_baseline()
        
        # Run quantum methods
        quantum_results = self.run_quantum_risk()
        
        # Compare results
        comparison_df = self.compare_results(classical_results, quantum_results)
        
        # Compile results
        results = {
            "experiment": "quantum_vs_classical_risk",
            "timestamp": datetime.now().isoformat(),
            "symbols": self.symbols,
            "confidence_level": self.confidence_level,
            "random_seed": self.random_seed,
            "classical_results": classical_results,
            "quantum_results": quantum_results,
            "comparison": comparison_df.to_dict("records")
        }
        
        return results


if __name__ == "__main__":
    # Run experiment
    symbols = ["RELIANCE", "TCS", "HDFCBANK", "HDFCBANK", "ICICIBANK"]
    
    experiment = RiskBenchmarkExperiment(
        symbols=symbols,
        confidence_level=0.95,
        random_seed=42
    )
    
    results = experiment.run_full_experiment()
    
    # Save results
    output_dir = Path(__file__).parent.parent.parent / "results-and-reporting"
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "experiment_01_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Results saved to:", output_file)
    print("=" * 60)
    
    # Print comparison table
    comparison_df = pd.DataFrame(results["comparison"])
    print("\nComparison Table:")
    print(comparison_df.to_string(index=False))

