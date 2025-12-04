"""
Main script to run all research experiments.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from research.experiment_01_quantum_vs_classical_risk import RiskBenchmarkExperiment
from research.experiment_02_quantum_portfolio_optimization import PortfolioOptimizationBenchmark


def main():
    """Run all research experiments."""
    print("=" * 70)
    print("Hybrid Quantum-Classical Portfolio & Risk Engine")
    print("Research Experiments")
    print("=" * 70)
    
    # Get available symbols
    from data.preprocessing_pipeline import PreprocessingPipeline
    pipeline = PreprocessingPipeline()
    all_symbols = pipeline.loader.get_available_symbols()
    print(f"\nAvailable symbols: {all_symbols}")
    
    # Experiment 1: Quantum vs Classical Risk Assessment
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Quantum vs Classical Risk Assessment")
    print("=" * 70)
    
    symbols_exp1 = all_symbols[:5]  # Use first 5 symbols
    experiment1 = RiskBenchmarkExperiment(
        symbols=symbols_exp1,
        confidence_level=0.95,
        random_seed=42
    )
    
    results1 = experiment1.run_full_experiment()
    
    # Save results
    output_dir = Path(__file__).parent / "src" / "results-and-reporting"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    output_file1 = output_dir / "experiment_01_results.json"
    with open(output_file1, "w") as f:
        json.dump(results1, f, indent=2)
    print(f"\nResults saved to: {output_file1}")
    
    # Experiment 2: Quantum Portfolio Optimization
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Quantum-Enhanced Portfolio Optimization")
    print("=" * 70)
    
    symbols_exp2 = all_symbols[:5]  # Use first 5 symbols
    experiment2 = PortfolioOptimizationBenchmark(
        symbols=symbols_exp2,
        risk_aversion=2.0,
        random_seed=42
    )
    
    results2 = experiment2.run_full_experiment()
    
    # Save results
    output_file2 = output_dir / "experiment_02_results.json"
    with open(output_file2, "w") as f:
        json.dump(results2, f, indent=2)
    print(f"\nResults saved to: {output_file2}")
    
    # Generate visualizations
    print("\n" + "=" * 70)
    print("Generating visualizations...")
    print("=" * 70)
    
    try:
        # Import visualization module (handle hyphen in directory name)
        import importlib.util
        viz_path = Path(__file__).parent / "src" / "results-and-reporting" / "visualize_results.py"
        spec = importlib.util.spec_from_file_location("visualize_results", viz_path)
        viz_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(viz_module)
        
        viz_module.plot_risk_comparison(str(output_file1), output_dir=str(output_dir))
        viz_module.plot_optimization_comparison(str(output_file2), output_dir=str(output_dir))
        viz_module.plot_scalability(str(output_file2), output_dir=str(output_dir))
        
        print("\nVisualizations generated successfully!")
    except Exception as e:
        print(f"\nWarning: Could not generate visualizations: {e}")
        print("You can run visualize_results.py separately.")
    
    print("\n" + "=" * 70)
    print("All experiments completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()

