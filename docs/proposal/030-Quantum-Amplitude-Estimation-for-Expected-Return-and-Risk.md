## 1. Methodology for Implementation

### 1.1 Overview

This tool extends QAE-based estimation to compute both expected return and risk (variance/CVaR proxy) simultaneously by encoding multiple payoff functions into quantum amplitudes. The goal is to accelerate joint estimation of mean and risk metrics used in portfolio optimization and risk gating.

Two strategies:

1. **Multi-ancilla QAE** – encode expected return in one ancilla and squared return (variance proxy) or tail indicator (risk proxy) in another.
2. **Sequential QAE** – reuse prepared state and run QAE with different payoff oracles (mean, variance, tail loss).

---

### 1.2 Inputs

| Input                 | Description                                      |
| --------------------- | ------------------------------------------------ |
| Scenario distribution | Same as expected-return QAE                      |
| Payoff functions      | f_mean(x), f_risk(x) (e.g., squared loss, tail)  |
| Qubit registers       | Scenario qubits + ancilla qubits per payoff      |
| Precision targets     | ε_mean, ε_risk controlling iterations            |

Risk metrics supported:

- Variance / standard deviation (E[(R-μ)²])
- Tail probability (indicator R < threshold)
- CVaR component (max(0, threshold - R))

---

### 1.3 Algorithm Steps

1. **Unified state preparation**
   - Load scenario distribution once.
   - Implement controlled rotations for each payoff (mean, risk).

2. **Amplitude estimation**
   - Use Iterative QAE or Maximum Likelihood QAE for each ancilla.
   - If multi-ancilla approach used, measure joint amplitudes to extract metrics.

3. **Post-processing**
   - Convert amplitudes to expected return and risk metric via scaling.
   - Compute derived statistics (variance = E[R²] - μ², CVaR = threshold + tail amplitude).

4. **Validation**
   - Compare vs classical Monte Carlo estimates.
   - Record precision, runtime, number of oracle calls.

---

### 1.4 Integration Points

- **Optimizer**: supply both μ and σ (or CVaR) to Markowitz/QMV constraints.
- **Risk module**: feed tail estimates directly into VaR/CVaR gating.
- **Experiment tracker**: store results to evaluate quantum vs classical speed/accuracy.

---

## 2. Literature & Motivation

- Stamatopoulos et al. (2020) – Quantum algorithms for Monte Carlo finance. ([arXiv][1])
- Montanaro (2015) – Quantum speedup of Monte Carlo methods. ([Proc. Roy. Soc. A][2])
- Zhao et al. (2021) – Simultaneous quantum estimation of multiple moments. ([arXiv][3])

Reasoning:

- Many workflows need both expected return and volatility tail metrics; computing them with shared quantum state reduces overhead.
- Demonstrates modular quantum subroutines feeding directly into classical optimization and risk pipelines.

---

## 3. Example Skeleton

```python
state_prep = build_state_preparation()
mean_oracle = build_payoff_oracle(payoff='mean')
risk_oracle = build_payoff_oracle(payoff='tail')

expected_return = run_qae(state_prep, mean_oracle, epsilon=0.01)
tail_expectation = run_qae(state_prep, risk_oracle, epsilon=0.01)
variance = compute_variance(expected_return, tail_expectation)
```

---

## 4. Research Directions

1. **Correlated estimation** – exploit covariance between mean and risk payoffs to reduce measurement counts.
2. **Adaptive precision** – allocate more QAE iterations to whichever metric has higher sensitivity for current optimizer iteration.
3. **Error mitigation** – analyze impact of correlated noise on multi-ancilla encodings.

[1]: https://arxiv.org/abs/2009.03842
[2]: https://royalsocietypublishing.org/doi/10.1098/rspa.2015.0301
[3]: https://arxiv.org/abs/2103.09230

