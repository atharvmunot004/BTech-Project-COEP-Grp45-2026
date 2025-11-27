## 1. Methodology for Implementation

### 1.1 Overview

Quantum k-Means leverages quantum distance estimation and amplitude encoding to accelerate clustering. In practice, hybrid algorithms (e.g., Lloyd’s algorithm with quantum subroutines) are used: quantum circuits compute distances or kernel values, while cluster assignments still run classically. This tool extends the PCA + K-Means pipeline by replacing or augmenting distance computations with quantum routines.

Targets:

- Cluster assets/regimes in feature space.
- Provide benchmark vs classical clustering.
- Feed cluster labels to regime detection and optimizer modules.

---

### 1.2 Inputs

| Input        | Description                                       |
| ------------ | ------------------------------------------------- |
| Feature matrix | Same as PCA + K-Means (normalized features)     |
| n_clusters   | Number of clusters                               |
| Encoding     | Amplitude or angle embedding for distance calc    |
| Backend      | Quantum simulator/hardware                        |

---

### 1.3 Hybrid Algorithm Steps

1. **Data normalization & encoding**
   - Scale features to unit norm (for amplitude encoding).
   - Map vectors to quantum states |x⟩.

2. **Quantum distance estimation**
   - Use swap test or amplitude estimation to compute distance between |x⟩ and cluster centroid states.
   - Centroids encoded as quantum states (updated classically each iteration).

3. **Cluster assignment**
   - Use quantum-computed distances in classical argmin step.
   - Update centroids classically; re-encode for next iteration.

4. **Termination**
   - Repeat until centroid shift < tolerance or max iterations reached.

---

### 1.4 Integration

- **Regime detection**: quantum clusters compared to classical ones.
- **Optimizer**: use cluster outputs for asset grouping/bucketing.
- **Research**: evaluate quantum overhead vs potential accuracy/runtime gains.

---

## 2. Literature & Motivation

- Lloyd et al. (2013) – Quantum algorithms for clustering via distance estimation. ([arXiv][1])
- Kerenidis et al. (2019) – q-means: quantum algorithm for unsupervised learning. ([arXiv][2])
- Schuld (2020) – Practical limitations / hybrid approaches. ([Quantum Inf. Process.][3])

Reasoning:

- Clustering is central to regime detection; quantum subroutines offer potential speedups for high-dimensional computations.
- Aligns with architecture requirement “Quantum k-Means / Clustering”.

---

## 3. Example Workflow

```python
features = normalize(features)
quantum_encoder = build_encoder(features_dim, n_qubits)
cluster_states = initialize_quantum_centroids(k)

for iteration in range(max_iter):
    distances = estimate_quantum_distances(features, cluster_states, backend)
    assignments = classical_argmin(distances)
    cluster_states = update_centroids(assignments, features, encoder)
```

---

## 4. Research Directions

1. **Kernelized quantum clustering** – combine QSVM kernels with clustering.
2. **Noise-aware swap tests** – evaluate error mitigation strategies for distance estimation.
3. **Streaming quantum clustering** – update centroids incrementally as new data arrives.

[1]: https://arxiv.org/abs/1304.7827
[2]: https://arxiv.org/abs/1904.02260
[3]: https://link.springer.com/article/10.1007/s11128-020-02974-x

