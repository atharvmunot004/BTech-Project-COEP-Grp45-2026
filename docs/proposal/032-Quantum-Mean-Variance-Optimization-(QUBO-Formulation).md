## 1. Methodology for Implementation

### 1.1 Overview

This variant of QMV focuses explicitly on the **QUBO/Ising formulation** suitable for quantum annealers (D-Wave) or QAOA hardware. Portfolio weights are discretized (e.g., binary fractions or asset inclusion decisions), allowing the mean-variance objective plus constraints to be written as:

```
minimize  xᵀ Q x + cᵀ x + constant
```

where x is binary and Q encodes risk/return trade-offs and penalties.

---

### 1.2 Discretization Scheme

1. **Binary indicator approach**
   - x_i ∈ {0,1} indicates whether asset i is included.
   - Constraint on number of assets or budget via penalty terms.

2. **Multi-bit weight encoding**
   - Represent weight w_i as sum of binary variables (e.g., 4-bit precision).
   - Allows approximate continuous weights at cost of more qubits.

Penalty formulation:

- Budget constraint: P_budget (Σ w_i − 1)².
- Long-only: implicit in binary encoding.
- Cardinality: P_card (Σ x_i − k)².

---

### 1.3 Building the QUBO Matrix

1. Start from objective `max wᵀ μ − λ wᵀ Σ w`.
2. Convert to minimization by negating objective.
3. Expand quadratic terms into binary variables according to encoding.
4. Add penalty matrices for constraints.
5. Scale coefficients to match hardware limits (typically [−2, 2]).

---

### 1.4 Quantum Solvers

- **Quantum annealing (D-Wave Advantage)**: submit QUBO directly, obtain low-energy solutions via anneals.
- **QAOA**: implement cost Hamiltonian from QUBO matrix, mix with driver Hamiltonian, optimize angles.
- **Classical post-processing**: take measurement samples, decode to weights, evaluate objective.

---

### 1.5 Integration Workflow

1. Generate QUBO from current μ, Σ, constraints.
2. Submit to chosen quantum solver (annealer or QAOA service).
3. Collect candidate portfolios, decode, evaluate classically.
4. Select best feasible solution and feed to execution pipeline.
5. Log solver metadata (anneal time, success probability, parameter angles).

---

## 2. Literature & References

- Venturelli et al. (2019) – Quantum optimization for finance with QUBO mappings. ([Front. Phys.][1])
- Rosenberg et al. (2016) – Portfolio rebalancing on D-Wave. ([arXiv][2])
- Hodson et al. (2019) – Portfolio selection as QUBO for annealers. ([arXiv][3])

Motivation:

- Direct compatibility with existing quantum annealing hardware.
- Provides benchmark for discrete/limited-weight portfolios.
- Aligns with architecture’s “QMV / QUBO Formulation” requirement.

---

## 3. Example (D-Wave Ocean)

```python
from dwave.system import DWaveSampler, EmbeddingComposite
from dimod import BinaryQuadraticModel

Q, linear = build_qubo(mu, Sigma, lambda_, penalties)
bqm = BinaryQuadraticModel(linear, Q, offset, vartype='BINARY')

sampler = EmbeddingComposite(DWaveSampler())
sampleset = sampler.sample(bqm, num_reads=1000)
best = sampleset.first.sample
weights = decode_weights(best)
```

---

## 4. Research Directions

1. **Precision vs qubit trade-off** – study how many bits per weight deliver acceptable performance before qubit requirements explode.
2. **Penalty calibration** – automated tuning (cross-validation, Bayesian optimization) of penalty strengths for constraints.
3. **Hybrid solvers** – use quantum annealer output as seeds for classical local search to boost accuracy.

[1]: https://www.frontiersin.org/articles/10.3389/fphy.2019.00031/full
[2]: https://arxiv.org/abs/1604.05718
[3]: https://arxiv.org/abs/1911.06259

