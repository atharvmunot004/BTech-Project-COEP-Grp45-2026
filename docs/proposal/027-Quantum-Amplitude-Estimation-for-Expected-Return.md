## 1. Methodology for Implementation

### 1.1 Overview

Quantum Amplitude Estimation (QAE) can accelerate Monte Carlo estimation of expectations by reducing sample complexity from O(1/ε²) to O(1/ε). Applied to expected return estimation, QAE encodes payoff distributions into amplitudes and uses quantum phase estimation to estimate mean return more efficiently. This tool complements classical Monte Carlo expected return estimators and serves as a building block for quantum portfolio optimization.

Use-cases:

- Rapid estimation of expected returns for large scenario sets.
- Subroutine inside quantum portfolio optimizers (QMV/QAOA).
- Benchmark for quantum advantage in expectation estimation.

---

### 1.2 Inputs & Encoding

| Input                  | Description                                                 |
| ---------------------- | ----------------------------------------------------------- |
| Scenario distribution  | Probability distribution over returns (from Monte Carlo)    |
| Payoff function        | Maps scenario to return contribution                       |
| Qubits (n_q)           | Number of qubits for state encoding and ancilla            |
| Iterations / shots     | Controls precision vs runtime                              |

Encoding workflow:

1. Prepare superposition over scenarios with amplitudes √p_i.
2. Encode payoff/return into ancilla amplitude via controlled rotations.
3. Apply QAE algorithm (standard, iterative, maximum likelihood variants).

---

### 1.3 Algorithm Steps

1. **State preparation**
   - Build `A |0⟩ = Σ √p_i |i⟩ |0⟩`.
   - Controlled rotation encodes return value into ancilla: amplitude proportional to expected return.

2. **Amplitude estimation**
   - Apply QAE (e.g., Iterative QAE to reduce depth).
   - Measure to obtain amplitude estimate a ≈ E[return].

3. **Expected return computation**
   - Convert amplitude to expected return using scaling constants from encoding.
   - Provide confidence intervals based on QAE variant used.

4. **Validation**
   - Compare vs classical Monte Carlo estimates (accuracy, runtime).
   - Log number of oracle calls, circuit depth, variance.

---

### 1.4 Integration Points

- **Portfolio optimizer**: supply expected returns for Markowitz/QMV constraints.
- **Risk module**: feed expected returns into VaR/CVaR computations.
- **Tool selector**: decide when quantum estimation outperforms classical (large scenario counts).

---

## 2. Literature & Motivation

- Brassard et al. (2002) – Original amplitude estimation algorithm. ([Contemporary Mathematics][1])
- Montanaro (2015) – Quantum speedup of Monte Carlo methods. ([Proc. Roy. Soc. A][2])
- Stamatopoulos et al. (2020) – Quantum algorithms for Monte Carlo pricing. ([arXiv][3])

Why relevant:

- Expected return estimation is core input for portfolio construction; QAE offers theoretical speedup.
- Acts as stepping stone toward quantum mean-variance optimization and QAE for CVaR.
- Provides quantitative evidence of quantum advantage boundaries in your stack.

---

## 3. Example (Iterative QAE Skeleton)

```python
from qiskit.algorithms import IterativeAmplitudeEstimation
from qiskit.circuit.library import StatePreparation, ZFeatureMap

state_preparation = build_return_loader(distribution_params)
grover_operator = build_grover_operator(state_preparation)

iae = IterativeAmplitudeEstimation(epsilon_target=0.01, alpha=0.05,
                                   state_preparation=state_preparation,
                                   grover_operator=grover_operator,
                                   objective_qubits=[ancilla_idx])
result = iae.estimate()
expected_return = scale_factor * result.estimation
confidence_interval = (scale_factor * result.confidence_interval[0],
                       scale_factor * result.confidence_interval[1])
```

---

## 4. Research Directions

1. **Maximum Likelihood QAE** – further reduces circuit depth while retaining quadratic speedup.
2. **Hybrid estimators** – blend classical Monte Carlo for coarse estimation with QAE refinement.
3. **Error mitigation** – study zero-noise extrapolation and probabilistic error cancellation to stabilize hardware runs.

[1]: https://arxiv.org/abs/quant-ph/0005055
[2]: https://royalsocietypublishing.org/doi/10.1098/rspa.2015.0301
[3]: https://arxiv.org/abs/2006.11382

