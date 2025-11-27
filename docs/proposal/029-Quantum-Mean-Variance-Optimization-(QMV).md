## 1. Methodology for Implementation

### 1.1 Overview

Quantum Mean-Variance Optimization (QMV) seeks to solve the classical Markowitz problem on quantum hardware by encoding portfolio weights into quantum states and using quantum solvers (QAOA, VQE, amplitude estimation) to optimize the quadratic objective. In your architecture, QMV provides a **quantum counterpart to Markowitz**, enabling comparative studies of solution quality, runtime, and scalability.

Problem statement:

```
maximize   wᵀ μ - λ wᵀ Σ w
subject to Σ w_i = 1, w_i ≥ 0 (or binary constraints for discrete weights)
```

---

### 1.2 Formulations

1. **Binary encoding + QAOA**
   - Discretize weights into binary variables.
   - Map objective to Ising Hamiltonian.
   - Use QAOA to approximate ground state representing optimal allocation.

2. **Amplitude estimation approach**
   - Use QAE to estimate expected return and variance within optimization loop.
   - Employ iterative quantum search to satisfy constraints.

3. **Variational continuous encoding**
   - Encode weights via parameterized rotations constrained to simplex.
   - Optimize expectation of objective using VQE.

---

### 1.3 Inputs

| Input          | Description                          |
| -------------- | ------------------------------------ |
| Expected returns μ | From forecasting modules (ARIMA/LSTM/Q-LSTM) |
| Covariance Σ   | From risk module (historical, GARCH) |
| Risk aversion λ| Policy parameter                     |
| Budget/constraints | Long-only, bounds, cardinality   |

Preprocessing:

- Scale μ and Σ to fit energy scales used in Hamiltonian.
- If using binary encoding, determine granularity per asset.

---

### 1.4 QAOA Workflow

1. **Quadratic Unconstrained Binary Optimization (QUBO)**
   - Convert objective + constraints into QUBO matrix Q.
   - Translate QUBO to Ising Hamiltonian H.

2. **QAOA circuit**
   - Initialize equal superposition over binary weight states.
   - Alternate cost unitary `exp(-i γ H)` and mixer unitary `exp(-i β H_mix)`.
   - Optimize angles (γ, β) using classical optimizer.

3. **Measurement & decoding**
   - Sample bitstrings, decode to portfolio weights.
   - Evaluate objective classically; keep best solution.

---

### 1.5 Integration

- **Optimizer service**: expose QMV as alternative to Markowitz solver.
- **Tool selector**: choose QMV for small/medium asset sets where quantum encoding feasible.
- **Benchmark harness**: compare solution quality vs classical MIQP solver.

---

## 2. Literature & Motivation

- Rebentrost et al. (2018) – Quantum algorithms for portfolio optimization. ([Phys. Rev. A][1])
- Barkoutsos et al. (2020) – QAOA for portfolio optimization on NISQ devices. ([Quantum][2])
- Hodson et al. (2019) – Portfolio rebalancing with QAOA. ([arXiv][3])

Why pursue QMV:

- Directly aligns with Markowitz baseline already implemented.
- Provides empirical evidence for/against near-term quantum advantage in optimization.
- Serves as testbed for other quantum optimization tools (annealing, QUBO solvers).

---

## 3. Example (QAOA-based Implementation Sketch)

```python
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.algorithms import QAOA
from qiskit.primitives import Estimator

qp = QuadraticProgram()
# add binary variables x_i representing discrete weight choices
for i in range(n_assets * levels):
    qp.binary_var(name=f"x_{i}")

qp.minimize(quadratic=Q_matrix)  # derived from objective + constraints

qaoa = QAOA(sampler=Estimator(), reps=2, optimizer=COBYLA())
optimizer = MinimumEigenOptimizer(qaoa)
result = optimizer.solve(qp)
weights = decode_binary_solution(result.x)
```

---

## 4. Research Extensions

1. **Constraint handling** – experiment with penalty factors, Lagrange multipliers, or slack qubits for soft constraints.
2. **Warm-start QAOA** – initialize angles using classical solutions to improve convergence.
3. **Quantum-inspired heuristics** – use measurements from QAOA to seed classical local search.

[1]: https://journals.aps.org/pra/abstract/10.1103/PhysRevA.98.022321
[2]: https://quantum-journal.org/papers/q-2020-07-06-291/
[3]: https://arxiv.org/abs/1911.06259

