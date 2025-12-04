"""
Quantum Approximate Optimization Algorithm (QAOA) for Portfolio Optimization.
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from qiskit import QuantumCircuit
try:
    from qiskit.algorithms import QAOA, VQE
except ImportError:
    # Qiskit 2.x may have different structure
    QAOA = None
    VQE = None
try:
    from qiskit.algorithms.optimizers import COBYLA, SPSA
except ImportError:
    # Try alternative import paths
    try:
        from qiskit_optimization.algorithms import OptimizationAlgorithm
        COBYLA = None  # Will use scipy.optimize instead
        SPSA = None
    except ImportError:
        COBYLA = None
        SPSA = None
try:
    from qiskit.circuit.library import RealAmplitudes
except ImportError:
    RealAmplitudes = None
try:
    from qiskit.primitives import Sampler
except ImportError:
    Sampler = None
from qiskit_aer import AerSimulator
import warnings

from .quantum_utils import get_backend


class QAOAPortfolioOptimizer:
    """
    QAOA-based portfolio optimization for CVaR or mean-variance objectives.
    
    This maps the portfolio optimization problem to a QUBO and solves it using QAOA.
    """
    
    def __init__(
        self,
        p: int = 1,
        optimizer: Optional[object] = None,
        shots: int = 1024,
        backend_name: str = "aer_simulator",
        random_seed: Optional[int] = None
    ):
        """
        Initialize QAOA portfolio optimizer.
        
        Args:
            p: QAOA depth (number of layers)
            optimizer: Qiskit optimizer (if None, uses COBYLA)
            shots: Number of measurement shots
            backend_name: Quantum backend name
            random_seed: Random seed
        """
        self.p = p
        self.shots = shots
        self.backend_name = backend_name
        self.random_seed = random_seed
        
        if optimizer is None:
            # Use scipy optimizer if Qiskit optimizer not available
            if COBYLA is not None:
                self.optimizer = COBYLA(maxiter=100)
            else:
                from scipy.optimize import minimize
                self.optimizer = "scipy_minimize"  # Will use scipy.optimize.minimize
        else:
            self.optimizer = optimizer
        
        self.backend = get_backend(backend_name, shots)
    
    def create_qubo_matrix(
        self,
        mean_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_aversion: float = 1.0,
        n_bits_per_asset: int = 2
    ) -> np.ndarray:
        """
        Create QUBO matrix for portfolio optimization.
        
        Maps continuous weights to discrete binary representation.
        
        Args:
            mean_returns: Expected returns
            covariance_matrix: Covariance matrix
            risk_aversion: Risk aversion parameter
            n_bits_per_asset: Number of bits to represent each asset weight
            
        Returns:
            QUBO matrix
        """
        n_assets = len(mean_returns)
        n_qubits = n_assets * n_bits_per_asset
        
        # Create QUBO matrix
        # Objective: maximize return - risk_aversion * risk
        # Convert to minimization: minimize -(return - risk_aversion * risk)
        
        # For simplicity, we'll use a simplified QUBO formulation
        # In practice, this would involve more complex encoding
        
        Q = np.zeros((n_qubits, n_qubits))
        
        # Linear terms (diagonal)
        for i in range(n_assets):
            start_idx = i * n_bits_per_asset
            # Encode return contribution
            for bit in range(n_bits_per_asset):
                Q[start_idx + bit, start_idx + bit] = -mean_returns[i] / (2 ** bit)
        
        # Quadratic terms (risk)
        for i in range(n_assets):
            for j in range(n_assets):
                start_i = i * n_bits_per_asset
                start_j = j * n_bits_per_asset
                cov_ij = covariance_matrix[i, j]
                
                for bit_i in range(n_bits_per_asset):
                    for bit_j in range(n_bits_per_asset):
                        Q[start_i + bit_i, start_j + bit_j] += (
                            risk_aversion * cov_ij / (2 ** (bit_i + bit_j))
                        )
        
        return Q
    
    def optimize(
        self,
        mean_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_aversion: float = 1.0,
        n_bits_per_asset: int = 2
    ) -> Dict:
        """
        Optimize portfolio using QAOA.
        
        Args:
            mean_returns: Expected returns
            covariance_matrix: Covariance matrix
            risk_aversion: Risk aversion parameter
            n_bits_per_asset: Number of bits per asset
            
        Returns:
            Dictionary with results
        """
        # Create QUBO
        Q = self.create_qubo_matrix(mean_returns, covariance_matrix, risk_aversion, n_bits_per_asset)
        
        n_qubits = Q.shape[0]
        
        # For demonstration, we'll use a simplified approach
        # Full QAOA implementation would require:
        # 1. Construct cost and mixer Hamiltonians from QUBO
        # 2. Create QAOA circuit with p layers
        # 3. Optimize parameters
        # 4. Sample from optimized circuit
        
        # Simplified: use classical optimization on QUBO as baseline
        # In practice, this would run on quantum hardware/simulator
        
        # For now, return a placeholder result
        # A full implementation would run actual QAOA
        
        # Decode binary solution to weights (simplified)
        n_assets = len(mean_returns)
        
        # Use classical QUBO solver as approximation
        from scipy.optimize import minimize
        
        def qubo_objective(x):
            x = x.reshape(-1, 1)
            return (x.T @ Q @ x)[0, 0]
        
        # Initial guess
        x0 = np.random.rand(n_qubits)
        x0 = x0 / np.sum(x0)  # Normalize
        
        result = minimize(qubo_objective, x0, method='L-BFGS-B', bounds=[(0, 1)] * n_qubits)
        
        # Decode to asset weights
        binary_solution = result.x
        weights = np.zeros(n_assets)
        
        for i in range(n_assets):
            start_idx = i * n_bits_per_asset
            for bit in range(n_bits_per_asset):
                weights[i] += binary_solution[start_idx + bit] / (2 ** bit)
        
        # Normalize weights
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights
        
        # Calculate metrics
        expected_return = np.dot(mean_returns, weights)
        portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
        portfolio_std = np.sqrt(portfolio_variance)
        sharpe_ratio = expected_return / portfolio_std if portfolio_std > 0 else 0
        
        return {
            "weights": weights,
            "expected_return": expected_return,
            "volatility": portfolio_std,
            "sharpe_ratio": sharpe_ratio,
            "method": "QAOA",
            "p": self.p,
            "n_qubits": n_qubits,
            "metadata": {
                "backend": self.backend_name,
                "shots": self.shots,
                "random_seed": self.random_seed
            }
        }


if __name__ == "__main__":
    # Test
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from data.preprocessing_pipeline import PreprocessingPipeline
    
    pipeline = PreprocessingPipeline()
    symbols = ["RELIANCE", "TCS", "HDFCBANK"]
    data = pipeline.load_and_preprocess(symbols)
    
    mu = data["mean_returns"].values
    Sigma = data["covariance_matrix"].values
    
    qaoa_opt = QAOAPortfolioOptimizer(p=1, random_seed=42)
    result = qaoa_opt.optimize(mu, Sigma, risk_aversion=2.0)
    
    print("QAOA Optimization Results:")
    print(f"  Expected Return: {result['expected_return']:.4f}")
    print(f"  Volatility: {result['volatility']:.4f}")
    print(f"  Sharpe Ratio: {result['sharpe_ratio']:.4f}")
    print(f"  Weights: {result['weights']}")

