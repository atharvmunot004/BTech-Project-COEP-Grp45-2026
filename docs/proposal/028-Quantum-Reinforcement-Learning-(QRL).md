## 1. Methodology for Implementation

### 1.1 Overview

Quantum Reinforcement Learning (QRL) integrates quantum subroutines (quantum policy representations, amplitude estimation, quantum value iteration) into RL workflows. In your architecture, QRL extends the classical RL trading agent by embedding quantum circuits for policy/value estimation, potentially reducing sample complexity or enabling richer policy classes.

Approach variants:

- **Quantum policy networks**: Parameterized quantum circuits produce action probabilities.
- **Quantum value estimation**: Use QAE to estimate state/action values more efficiently.
- **Hybrid QRL**: Classical agent augmented with quantum kernels or generators.

---

### 1.2 Environment & Inputs

Same environment as classical RL, plus:

| Component         | Description                                                       |
| ----------------- | ----------------------------------------------------------------- |
| Quantum policy PQC| Circuit mapping observations to action amplitudes                 |
| Backend           | Simulator/hardware for PQC evaluation                             |
| Shots             | Measurements per policy evaluation (trade-off accuracy/runtime)   |
| Classical buffer  | Replay memory for hybrid training                                 |

---

### 1.3 Algorithm Examples

1. **Variational Quantum Policy Gradient**
   - Observations encoded via amplitude or angle embedding.
   - PQC output measured to obtain action probabilities.
   - Policy gradients computed via parameter-shift rule.

2. **Quantum Q-learning**
   - Use QAE to estimate expected return of action.
   - Update Q-table/policy with quantum-estimated values.

3. **Hybrid actor-critic**
   - Actor is quantum circuit, critic remains classical neural network.

---

### 1.4 Training Workflow

1. **State encoding** – compress observation to qubit-count-friendly vector (PCA, autoencoder).
2. **Policy circuit design** – choose ansatz depth/entanglement; ensure hardware-efficient layout.
3. **Gradient evaluation** – parameter-shift or finite difference using quantum backend.
4. **Learning loop** – similar to PPO/A2C but replacing actor with quantum policy.
5. **Evaluation** – compare profitability, risk metrics vs classical RL agent.

---

### 1.5 Integration Points

- **Tool orchestrator**: choose between classical RL and QRL based on regime/performance.
- **Quantum resource manager**: allocate backend time/qubits for policy evaluations.
- **Risk guardrails**: same risk constraints as classical RL; ensure outputs obey limits.

---

## 2. Literature & Motivation

- Jerbi et al. (2021) – Quantum RL with quantum circuits. ([Quantum Sci. Technol.][1])
- Skolik et al. (2022) – Hybrid quantum-classical policy gradient methods. ([Quantum][2])
- Lockwood & Meier (2020) – Quantum-enhanced SARSA. ([arXiv][3])

Relevance:

- Provides platform to explore quantum advantage in decision-making.
- Works directly with existing RL environment; drop-in replacement for policy components.
- Supports research goal of comparing classical vs quantum versions of same tool.

---

## 3. Example (Quantum Policy Gradient)

```python
import pennylane as qml
import torch
from torch.optim import Adam

dev = qml.device('default.qubit', wires=n_qubits, shots=1024)

@qml.qnode(dev, interface='torch')
def policy_circuit(obs, weights):
    qml.templates.AngleEmbedding(obs, wires=range(n_qubits))
    qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_actions_qubits)]

class QuantumPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight_shapes = {"weights": (layers, n_qubits)}
        self.qlayer = qml.qnn.TorchLayer(policy_circuit, self.weight_shapes)

    def forward(self, obs):
        logits = self.qlayer(obs)
        return torch.softmax(logits, dim=-1)

policy = QuantumPolicy()
optimizer = Adam(policy.parameters(), lr=1e-3)
```

---

## 4. Research Directions

1. **QAE-accelerated critics** – apply amplitude estimation to critic value functions.
2. **Quantum exploration strategies** – use quantum superposition to explore action space.
3. **Hardware benchmarking** – quantify fidelity requirements (shots, noise mitigation) for stable trading performance.

[1]: https://iopscience.iop.org/article/10.1088/2058-9565/ac1d78
[2]: https://quantum-journal.org/papers/q-2022-02-24-651/
[3]: https://arxiv.org/abs/2008.02823

