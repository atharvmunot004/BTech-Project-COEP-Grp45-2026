"""
Research Experiment 2: Quantum-Enhanced Portfolio Optimization

Compare QAOA/QMV against classical Markowitz and CVaR optimization.
Evaluate scalability with number of assets and solution quality.
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
from classical.portfolio.markowitz import markowitz_optimization
from classical.portfolio.cvar_lp import cvar_optimization
from quantum.qaoa_opt import QAOAPortfolioOptimizer


class PortfolioOptimizationBenchmark:
    """Benchmark quantum vs classical portfolio optimization."""
    
    def __init__(
        self,
        symbols: list,
        risk_aversion: float = 2.0,
        random_seed: int = 42
    ):
        """
        Initialize experiment.
        
        Args:
            symbols: List of stock symbols
            risk_aversion: Risk aversion parameter
            random_seed: Random seed
        """
        self.symbols = symbols
        self.risk_aversion = risk_aversion
        self.random_seed = random_seed
        
        # Load and preprocess data
        print("Loading and preprocessing data...")
        self.pipeline = PreprocessingPipeline()
        self.data = self.pipeline.load_and_preprocess(symbols)
        
        self.mu = self.data["mean_returns"].values
        self.Sigma = self.data["covariance_matrix"].values
        self.returns = self.data["returns_df"].values
    
    def run_classical_optimization(self) -> dict:
        """Run classical portfolio optimization methods."""
        print("\n=== Running Classical Portfolio Optimization ===")
        
        results = {}
        
        # Markowitz Mean-Variance
        print("Markowitz Mean-Variance Optimization...")
        start_time = time.time()
        markowitz_result = markowitz_optimization(
            self.mu,
            self.Sigma,
            risk_aversion=self.risk_aversion
        )
        results["markowitz"] = {
            "weights": markowitz_result["weights"].tolist(),
            "expected_return": float(markowitz_result["expected_return"]),
            "volatility": float(markowitz_result["volatility"]),
            "sharpe_ratio": float(markowitz_result["sharpe_ratio"]),
            "runtime": time.time() - start_time
        }
        print(f"  Expected Return: {markowitz_result['expected_return']:.4f}")
        print(f"  Volatility: {markowitz_result['volatility']:.4f}")
        print(f"  Sharpe Ratio: {markowitz_result['sharpe_ratio']:.4f}")
        
        # CVaR Optimization
        print("\nCVaR Optimization...")
        start_time = time.time()
        cvar_result = cvar_optimization(
            self.returns,
            confidence_level=0.95
        )
        results["cvar"] = {
            "weights": cvar_result["weights"].tolist(),
            "cvar": float(cvar_result["cvar"]),
            "expected_return": float(cvar_result["expected_return"]),
            "volatility": float(cvar_result["volatility"]),
            "sharpe_ratio": float(cvar_result["sharpe_ratio"]),
            "runtime": time.time() - start_time
        }
        print(f"  CVaR: {cvar_result['cvar']:.4f}")
        print(f"  Expected Return: {cvar_result['expected_return']:.4f}")
        print(f"  Sharpe Ratio: {cvar_result['sharpe_ratio']:.4f}")
        
        return results
    
    def run_quantum_optimization(self, p_values: list = [1, 2, 3]) -> dict:
        """Run quantum portfolio optimization methods."""
        print("\n=== Running Quantum Portfolio Optimization ===")
        
        results = {}
        
        # QAOA with different depths
        for p in p_values:
            print(f"\nQAOA with p={p}...")
            start_time = time.time()
            
            qaoa_optimizer = QAOAPortfolioOptimizer(
                p=p,
                random_seed=self.random_seed
            )
            
            qaoa_result = qaoa_optimizer.optimize(
                self.mu,
                self.Sigma,
                risk_aversion=self.risk_aversion
            )
            
            runtime = time.time() - start_time
            
            results[f"qaoa_p_{p}"] = {
                "weights": qaoa_result["weights"].tolist(),
                "expected_return": float(qaoa_result["expected_return"]),
                "volatility": float(qaoa_result["volatility"]),
                "sharpe_ratio": float(qaoa_result["sharpe_ratio"]),
                "runtime": runtime,
                "p": p,
                "n_qubits": qaoa_result["n_qubits"]
            }
            
            print(f"  Expected Return: {qaoa_result['expected_return']:.4f}")
            print(f"  Volatility: {qaoa_result['volatility']:.4f}")
            print(f"  Sharpe Ratio: {qaoa_result['sharpe_ratio']:.4f}")
            print(f"  Runtime: {runtime:.2f}s")
        
        return results
    
    def compare_results(self, classical_results: dict, quantum_results: dict) -> pd.DataFrame:
        """Compare classical and quantum optimization results."""
        comparison_data = []
        
        # Classical methods
        comparison_data.append({
            "method": "Markowitz",
            "type": "classical",
            "expected_return": classical_results["markowitz"]["expected_return"],
            "volatility": classical_results["markowitz"]["volatility"],
            "sharpe_ratio": classical_results["markowitz"]["sharpe_ratio"],
            "runtime": classical_results["markowitz"]["runtime"]
        })
        
        comparison_data.append({
            "method": "CVaR LP",
            "type": "classical",
            "expected_return": classical_results["cvar"]["expected_return"],
            "volatility": classical_results["cvar"]["volatility"],
            "sharpe_ratio": classical_results["cvar"]["sharpe_ratio"],
            "runtime": classical_results["cvar"]["runtime"]
        })
        
        # Quantum methods
        for key, value in quantum_results.items():
            comparison_data.append({
                "method": f"QAOA (p={value['p']})",
                "type": "quantum",
                "expected_return": value["expected_return"],
                "volatility": value["volatility"],
                "sharpe_ratio": value["sharpe_ratio"],
                "runtime": value["runtime"],
                "n_qubits": value["n_qubits"]
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Calculate differences relative to Markowitz (baseline)
        baseline_sharpe = classical_results["markowitz"]["sharpe_ratio"]
        df["sharpe_diff"] = df["sharpe_ratio"] - baseline_sharpe
        df["sharpe_diff_pct"] = (df["sharpe_diff"] / baseline_sharpe) * 100
        
        return df
    
    def scalability_test(self, n_assets_list: list = [5, 10, 15]) -> dict:
        """Test scalability with different numbers of assets."""
        print("\n=== Scalability Test ===")
        
        all_symbols = self.pipeline.loader.get_available_symbols()
        scalability_results = {}
        
        for n_assets in n_assets_list:
            print(f"\nTesting with {n_assets} assets...")
            test_symbols = all_symbols[:n_assets]
            
            # Load data
            test_data = self.pipeline.load_and_preprocess(test_symbols)
            test_mu = test_data["mean_returns"].values
            test_Sigma = test_data["covariance_matrix"].values
            
            # Classical Markowitz
            start_time = time.time()
            markowitz_result = markowitz_optimization(
                test_mu, test_Sigma, risk_aversion=self.risk_aversion
            )
            classical_time = time.time() - start_time
            
            # Quantum QAOA
            start_time = time.time()
            qaoa_optimizer = QAOAPortfolioOptimizer(p=1, random_seed=self.random_seed)
            qaoa_result = qaoa_optimizer.optimize(test_mu, test_Sigma, risk_aversion=self.risk_aversion)
            quantum_time = time.time() - start_time
            
            scalability_results[f"n_{n_assets}"] = {
                "n_assets": n_assets,
                "classical_runtime": classical_time,
                "quantum_runtime": quantum_time,
                "speedup": classical_time / quantum_time if quantum_time > 0 else 0,
                "classical_sharpe": float(markowitz_result["sharpe_ratio"]),
                "quantum_sharpe": float(qaoa_result["sharpe_ratio"])
            }
            
            print(f"  Classical: {classical_time:.3f}s, Quantum: {quantum_time:.3f}s")
        
        return scalability_results
    
    def run_full_experiment(self) -> dict:
        """Run complete experiment."""
        print("=" * 60)
        print("Research Experiment 2: Quantum-Enhanced Portfolio Optimization")
        print("=" * 60)
        
        # Run classical optimization
        classical_results = self.run_classical_optimization()
        
        # Run quantum optimization
        quantum_results = self.run_quantum_optimization()
        
        # Compare results
        comparison_df = self.compare_results(classical_results, quantum_results)
        
        # Scalability test
        scalability_results = self.scalability_test()
        
        # Compile results
        results = {
            "experiment": "quantum_portfolio_optimization",
            "timestamp": datetime.now().isoformat(),
            "symbols": self.symbols,
            "risk_aversion": self.risk_aversion,
            "random_seed": self.random_seed,
            "classical_results": classical_results,
            "quantum_results": quantum_results,
            "comparison": comparison_df.to_dict("records"),
            "scalability": scalability_results
        }
        
        return results


if __name__ == "__main__":
    # Run experiment
    symbols = ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY"]
    
    experiment = PortfolioOptimizationBenchmark(
        symbols=symbols,
        risk_aversion=2.0,
        random_seed=42
    )
    
    results = experiment.run_full_experiment()
    
    # Save results
    output_dir = Path(__file__).parent.parent.parent / "results-and-reporting"
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "experiment_02_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Results saved to:", output_file)
    print("=" * 60)
    
    # Print comparison table
    comparison_df = pd.DataFrame(results["comparison"])
    print("\nComparison Table:")
    print(comparison_df.to_string(index=False))

