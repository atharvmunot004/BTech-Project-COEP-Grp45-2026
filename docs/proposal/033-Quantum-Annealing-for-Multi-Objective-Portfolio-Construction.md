## 1. Methodology for Implementation

### 1.1 Overview

Quantum annealing (QA) can solve multi-objective portfolio problems by encoding objectives (return, risk, transaction cost, ESG score, etc.) into a single QUBO with tunable weights. This tool targets D-Wave Advantage (or similar annealers) to explore multi-objective trade-offs efficiently.

Objective example:

```
maximize   α wᵀ μ − β wᵀ Σ w + γ ESG(w) − δ Cost(w)
subject to constraints
```

Converted to QUBO with penalty terms and solved via QA.

---

### 1.2 Inputs

| Input            | Description                                           |
| ---------------- | ----------------------------------------------------- |
| μ, Σ             | Expected returns and covariance matrix                |
| Additional factors | ESG scores, liquidity metrics, carbon exposure     |
| Objective weights| α, β, γ, δ controlling trade-offs                     |
| Constraints      | Budget, bounds, cardinality, sector caps              |

---

### 1.3 QUBO Construction

1. **Binary encoding** – choose representation (binary weights or inclusion indicators).
2. **Objective aggregation** – combine multiple objectives with weights into single quadratic form.
3. **Constraint penalties** – budget, cardinality, sector constraints encoded as squared penalties.
4. **Coefficient scaling** – adjust coefficients to annealer range [-2, 2].

---

### 1.4 Annealing Workflow

1. Build QUBO (linear + quadratic terms, offset).
2. Submit to D-Wave via Ocean SDK with chosen anneal schedule/num_reads.
3. Post-process samples (decode weights, enforce feasibility).
4. Compute objective components per solution; plot Pareto frontier.

Optional: run multiple objective weight settings to map trade-off surface.

---

### 1.5 Integration

- **Optimizer layer**: QA solver offered alongside classical solvers.
- **Research notebook**: visualize QA solutions vs classical MOEA (NSGA-II) results.
- **Execution**: selected QA solution passed to order generation pipeline.

---

## 2. Literature & Motivation

- Venturelli & Kondratyev (2019) – QA for market problems. ([Front. Phys.][1])
- Fung et al. (2021) – Multi-objective portfolio optimization on D-Wave. ([arXiv][2])
- Rosenberg et al. (2016) – Portfolio rebalancing via QA. ([arXiv][3])

Benefits:

- Native support for multiple objectives via weighted sum or lexicographic runs.
- Ability to explore combinatorial constraint-heavy portfolios.
- Demonstrates use of annealers in production-style workflow.

---

## 3. Example (Ocean SDK)

```python
from dimod import BinaryQuadraticModel
from dwave.system import DWaveSampler, EmbeddingComposite

linear, quadratic, offset = build_multi_objective_qubo(mu, Sigma, esg, costs, weights)
bqm = BinaryQuadraticModel(linear, quadratic, offset, vartype='BINARY')

sampler = EmbeddingComposite(DWaveSampler())
sampleset = sampler.sample(bqm, num_reads=2000, annealing_time=20)
best = sampleset.first.sample
weights = decode(best)
```

---

## 4. Research Directions

1. **Adaptive weighting** – adjust objective weights dynamically based on regime signals.
2. **Pareto sampling** – run QA multiple times with varied weights to approximate Pareto frontier.
3. **Quantum-classical hybrid** – use QA-generated solutions to seed classical multi-objective optimizers.

[1]: https://www.frontiersin.org/articles/10.3389/fphy.2019.00031/full
[2]: https://arxiv.org/abs/2103.15912
[3]: https://arxiv.org/abs/1604.05718

