"""
Utility functions for quantum algorithms.
"""
import numpy as np
from typing import List, Tuple, Optional
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
try:
    from qiskit.circuit.library import QFT
except ImportError:
    QFT = None
from qiskit_aer import AerSimulator
try:
    from qiskit.primitives import Sampler
except ImportError:
    Sampler = None
import qiskit


def create_amplitude_estimation_circuit(
    n_qubits: int,
    state_preparation: QuantumCircuit,
    objective_qubit: int,
    n_evaluation_qubits: int = 3
) -> QuantumCircuit:
    """
    Create a quantum amplitude estimation circuit.
    
    Args:
        n_qubits: Number of qubits for the state
        state_preparation: Circuit that prepares the state
        objective_qubit: Index of qubit encoding the objective
        n_evaluation_qubits: Number of qubits for amplitude estimation
        
    Returns:
        Complete QAE circuit
    """
    # Create registers
    qr_state = QuantumRegister(n_qubits, 'state')
    qr_objective = QuantumRegister(1, 'objective')
    qr_evaluation = QuantumRegister(n_evaluation_qubits, 'evaluation')
    cr = ClassicalRegister(n_evaluation_qubits, 'c')
    
    # Create circuit
    qc = QuantumCircuit(qr_state, qr_objective, qr_evaluation, cr)
    
    # Prepare state
    qc.compose(state_preparation, qubits=list(range(n_qubits)), inplace=True)
    
    # Amplitude estimation (simplified version)
    # In practice, this would use iterative amplitude estimation or QPE
    # QFT would be used here if available, but we use manual Hadamard gates instead
    for i in range(n_evaluation_qubits):
        # Apply controlled rotations
        qc.h(qr_evaluation[i])
        # Grover operator would go here
        qc.measure(qr_evaluation[i], cr[i])
    
    return qc


def discretize_distribution(
    samples: np.ndarray,
    n_qubits: int,
    bounds: Optional[Tuple[float, float]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discretize a continuous distribution for quantum encoding.
    
    Args:
        samples: Array of samples from the distribution
        n_qubits: Number of qubits for discretization (2^n_qubits bins)
        bounds: (min, max) bounds. If None, uses min/max of samples
        
    Returns:
        Tuple of (bin_edges, probabilities)
    """
    n_bins = 2 ** n_qubits
    
    if bounds is None:
        min_val = np.min(samples)
        max_val = np.max(samples)
    else:
        min_val, max_val = bounds
    
    # Create bins
    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    
    # Calculate probabilities
    hist, _ = np.histogram(samples, bins=bin_edges)
    probabilities = hist / np.sum(hist)
    
    return bin_edges, probabilities


def load_distribution(
    probabilities: np.ndarray,
    n_qubits: int
) -> QuantumCircuit:
    """
    Create a quantum circuit that loads a probability distribution.
    
    Args:
        probabilities: Probability distribution (must be length 2^n_qubits)
        n_qubits: Number of qubits
        
    Returns:
        Quantum circuit that prepares the state
    """
    if len(probabilities) != 2 ** n_qubits:
        raise ValueError(f"Probabilities length must be 2^{n_qubits} = {2**n_qubits}")
    
    qr = QuantumRegister(n_qubits, 'q')
    qc = QuantumCircuit(qr)
    
    # Normalize probabilities
    probabilities = probabilities / np.sum(probabilities)
    amplitudes = np.sqrt(probabilities)
    
    # Use initialize method (simplified - in practice would use more efficient methods)
    qc.initialize(amplitudes, qr)
    
    return qc


def get_backend(backend_name: str = "aer_simulator", shots: int = 1024):
    """
    Get a quantum backend.
    
    Args:
        backend_name: Name of the backend
        shots: Number of shots for measurement
        
    Returns:
        Backend instance
    """
    if backend_name == "aer_simulator":
        return AerSimulator()
    else:
        raise ValueError(f"Unknown backend: {backend_name}")


if __name__ == "__main__":
    # Test discretization
    samples = np.random.normal(0, 1, 1000)
    bin_edges, probs = discretize_distribution(samples, n_qubits=3)
    print(f"Bin edges: {bin_edges}")
    print(f"Probabilities: {probs}")
    print(f"Sum: {np.sum(probs)}")

