## 1. Methodology for Implementation

### 1.1 Overview

Quantum Long Short-Term Memory (Q-LSTM) combines classical recurrent architectures with parameterized quantum circuits (PQCs) to encode sequence information. Typical designs replace LSTM gates with quantum subcircuits or use quantum neurons to process hidden states. In your stack, Q-LSTM acts as a **quantum-enhanced sequence forecaster** for returns, volatility, or factor series, enabling comparison against classical LSTM.

Design patterns:

- **Hybrid layer**: Classical embedding → PQC gate → classical activation.
- **Full Q-LSTM unit**: Parameterized quantum gates emulate input/forget/output gates.
- **Variational circuit**: Sequence processed by repeated PQCs with shared parameters.

---

### 1.2 Inputs & Architecture

| Component         | Description                                            |
| ----------------- | ------------------------------------------------------ |
| Time-series data  | Returns, log-prices, volatility metrics                |
| Sequence length   | e.g., 32–64 timesteps (limited by noise)               |
| Feature dimension | Reduced via PCA/factors to match qubit count           |
| Qubit count       | Typically 4–12 qubits per quantum layer                |
| Quantum backend   | Simulator (Qiskit Aer) or hardware for small circuits  |

Architecture example:

1. Classical dense layer to project features → dimension matching qubits.
2. Quantum gate block (entangling rotations) representing LSTM gates.
3. Classical readout layer combining quantum outputs with previous hidden state.

---

### 1.3 Training Procedure

1. **Data preprocessing**
   - Normalize time-series.
   - Create rolling windows (sequence_length, forecast_horizon).

2. **Model definition**
   - Implement custom PyTorch/PennyLane module with quantum layer inside LSTM cell.
   - Choose PQC ansatz (hardware-efficient, IQP, etc.).

3. **Optimization**
   - Loss: MSE/MAE for regression, cross-entropy for classification.
   - Optimizer: Adam with learning rate 1e-3 to 1e-4.
   - Shots: trade-off between variance and runtime.

4. **Evaluation**
   - Compare forecasts vs classical LSTM baseline.
   - Track loss curves, generalization gap.
   - Perform ablation (remove quantum layer) to quantify contribution.

5. **Deployment**
   - Export classical + quantum weights.
   - Provide simulator endpoint for inference; schedule retrains.

---

### 1.4 Integration Points

- **Forecasting pipeline**: Provide Q-LSTM forecasts as alternative expected returns/volatility inputs.
- **Tool selector**: Choose between LSTM and Q-LSTM based on regime or performance.
- **Research logging**: Record backend, qubit count, circuit depth, training metrics for reproducibility.

---

## 2. Literature & Motivation

- Chen et al. (2020) – Hybrid quantum-classical LSTM networks. ([arXiv][1])
- Kyriienko et al. (2021) – Quantum recurrent neural networks. ([Quantum Sci. Technol.][2])
- Zlokapa et al. (2022) – Q-LSTM for financial time series forecasting. ([arXiv][3])

Motivation:

- Explore whether quantum sequence models capture nonlinear dependencies with fewer parameters.
- Provide benchmark for quantum ML on sequential financial data.
- Investigate robustness under regime changes when classical models overfit.

---

## 3. Example (PennyLane Hybrid Cell)

```python
import pennylane as qml
import torch
from torch import nn

dev = qml.device('default.qubit', wires=n_qubits, shots=1024)

@qml.qnode(dev, interface='torch')
def quantum_gate(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QLSTMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_qubits, n_layers):
        super().__init__()
        self.weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(quantum_gate, self.weight_shapes)
        self.classical = nn.Linear(n_qubits, hidden_dim)

    def forward(self, x):
        q_out = self.qlayer(x)
        return self.classical(q_out)
```

---

## 4. Research Directions

1. **Noise-aware training** – incorporate hardware noise models during training to improve transfer.
2. **Adaptive circuit depth** – adjust PQC depth based on detected regime volatility.
3. **Quantum attention** – combine Q-LSTM with quantum attention mechanisms for long-range dependencies.

[1]: https://arxiv.org/abs/2009.01783
[2]: https://iopscience.iop.org/article/10.1088/2058-9565/ac1ab9
[3]: https://arxiv.org/abs/2206.02426

