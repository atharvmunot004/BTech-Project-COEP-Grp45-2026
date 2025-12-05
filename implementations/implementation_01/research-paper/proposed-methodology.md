# **Proposed Methodology**

This study investigates the comparative performance of **Quantum Amplitude Estimation (QAE)** and **classical risk-estimation methods** for Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR) across a broad spectrum of financial portfolios. The methodology is divided into five major components: **data preprocessing**, **portfolio generation**, **classical risk estimation**, **quantum risk estimation**, and **evaluation via benchmarking and backtesting**. Together, these components enable a comprehensive assessment of accuracy, convergence behavior, computational efficiency, and robustness under real market conditions.

---

## **1. Data Collection and Preprocessing**

### **1.1 Dataset**

We use **10 years of daily OHLCV data** for **10 diversified Indian equities**, selected to represent multiple sectors (Technology, Banking, FMCG, Pharma, etc.). Data is sourced from Yahoo Finance/NSE and aligned across all trading dates.

### **1.2 Price Alignment**

* Missing values are treated using forward-fill or removed to maintain consistent dates.
* Adjusted Close prices are used to capture corporate actions.

### **1.3 Return Computation**

Daily **log returns** are computed as:
$$
r_{t,i} = \ln\left(\frac{P_{t,i}}{P_{t-1,i}}\right)
$$
for assets $i = 1,\dots,10$.

### **1.4 Statistical Inputs**

From the return matrix $R \in \mathbb{R}^{T \times 10}$, we compute:

* Mean return vector:  $\mu = \mathbb{E}[R]$
* Covariance matrix:  $\Sigma = \text{Cov}(R)$

These provide the classical distributional parameters and serve as the basis for both classical and quantum risk modeling.

---

## **2. Portfolio Generation**

To evaluate performance across a broad range of risk profiles, we generate **100,000 random portfolios**, varying both allocation patterns and concentration levels.

### **2.1 Portfolio Size Sampling**

For each portfolio $p$:

* A random number of assets $k_p$ is selected uniformly from $\{3,\dots,10\}$.
* $k_p$ distinct assets are sampled without replacement.

### **2.2 Weight Generation**

Weights are generated using a **Dirichlet distribution**:
$$
w_p \sim \text{Dirichlet}(\alpha \mathbf{1}_{k_p})
$$
with $\alpha = 1$, ensuring:

* $w_{p,i} \geq 0$
* $\sum_i w_{p,i} = 1$

This approach produces fully invested, long-only portfolios with realistic variability.

### **2.3 Portfolio Return Model**

For each portfolio:
$$
\mu_p = w_p^\top \mu, \quad
\sigma_p = \sqrt{ w_p^\top \Sigma w_p }
$$

These parameters are used for parametric VaR and serve as validation inputs for simulation-based methods.

---

## **3. Classical Risk Estimation Framework**

We implement three classical risk models to establish strong baselines against which quantum methods are evaluated:

---

### **3.1 Parametric (Variance–Covariance) VaR**

Assuming multivariate normal returns, 1-day VaR at confidence $c$ is:
$$
\text{VaR}_{p,c}^{\text{param}}
= V ( z_c \sigma_p - \mu_p )
$$

This method is extremely fast but relies on distributional assumptions.

---

### **3.2 Historical Simulation VaR**

Using trailing return windows (e.g., 250 days):

* Compute empirical portfolio losses.
* VaR is taken as the empirical quantile:
  $$
  \text{VaR}_{c}^{\text{hist}} = \text{Quantile}_{c}(L_{t})
  $$

This approach is non-parametric but limited by the size of available history.

---

### **3.3 Classical Monte Carlo VaR and CVaR**

We simulate portfolio returns as:
$$
R^{(i)} \sim \mathcal{N}(\mu, \Sigma)
$$
$$
L^{(i)} = -V (w_p^\top R^{(i)})
$$

For each portfolio, 100,000 sampled losses estimate:

* VaR: empirical quantile of $L^{(i)}$
* CVaR: conditional mean of losses beyond VaR

To provide a **ground truth VaR**, we additionally generate **10 million Monte Carlo samples**.
This yields an extremely accurate reference value:
$$
\text{VaR}_{p,c}^{\text{true}} = \text{Quantile}_{c}(L_{1...10M})
$$

This serves as the benchmark for both classical and quantum estimation error.

---

## **4. Quantum Risk Estimation Framework**

We apply **Quantum Amplitude Estimation (QAE)** and its variants to approximate tail probability distributions and risk measures.

---

### **4.1 Distribution Encoding**

Portfolio loss distributions are encoded into quantum amplitudes via:

* **Amplitude Encoding**, or
* **Quantum State Preparation using log-normal approximations**, or
* **QGAN-generated distributions** for more realistic non-normal behavior.

The encoded distribution represents:
$$
\Pr(L \leq x) \quad \text{in amplitude space}
$$

---

### **4.2 Oracle Construction**

We design a **loss oracle**:
$$
f(x) = \mathbb{I}(L(x) > \text{VaR}_{\text{threshold}})
$$
for probability estimation, and an extended oracle for CVaR:
$$
g(x) = \max(0, L(x) - \text{VaR})
$$

These oracles form the core of QAE circuits.

---

### **4.3 Amplitude Estimation Methods**

We implement and benchmark:

1. **Standard Quantum Amplitude Estimation (QAE)**
2. **Iterative Amplitude Estimation (IQAE)**
3. **Maximum-Likelihood Amplitude Estimation (MLAE)**

For each portfolio:

* Compute estimated VaR/CVaR.
* Record quantum resource costs:

  * number of qubits
  * circuit depth
  * number of oracle calls
  * 1-qubit and 2-qubit gate counts

---

### **4.4 Quantum Runtime Environment**

* Simulations performed on Qiskit Aer.
* Small-scale circuits validated on real quantum hardware (IBM Q) to assess noise sensitivity.
* Real-hardware results included as feasibility analysis.

---

## **5. Evaluation Methodology**

We evaluate classical vs quantum methods using two complementary approaches:

---

# **5.1 Benchmarking Against True VaR**

For each of the 100,000 portfolios:

### **Accuracy Metrics**

$$
\text{MAE} = \frac{1}{N} \sum_{p} |\hat{\text{VaR}}_{p} - \text{VaR}_{p}^{\text{true}}|
$$
$$
\text{RMSE} = \sqrt{\frac{1}{N}\sum_{p} (\hat{\text{VaR}}_{p} - \text{VaR}_{p}^{\text{true}})^2 }
$$
$$
\text{Relative Error} = \frac{|\hat{\text{VaR}}_{p} - \text{VaR}_{p}^{\text{true}}|}{\text{VaR}_{p}^{\text{true}}}
$$

### **Efficiency Metrics**

* Wall-clock runtime
* Error vs number of samples (MC)
* Error vs oracle calls (QAE)

### **Quantum Resource Metrics**

* Qubit count
* Circuit depth
* Number of Grover iterations
* Gate count

### **Statistical Metrics**

* Confidence interval width
* Kolmogorov–Smirnov distance from true distribution
* Tail stability at 95%, 99%, 99.5% VaR levels

This benchmarking reveals **quadratic convergence advantages** of QAE over classical Monte Carlo.

---

# **5.2 Real Market Backtesting**

To assess real-world applicability, we conduct a **rolling VaR backtest**:

### **Procedure**

1. For each day (t), estimate VaR using the previous 1-year window.
2. Observe the next-day realized profit/loss.
3. Record a **VaR exception** if:
   $$
   L_{t+1} > \text{VaR}_{t}
   $$
4. Compare exception frequency with theoretical expectation:

   * 95% VaR → 5% exceptions
   * 99% VaR → 1% exceptions

### **Backtesting Metrics**

* Exception rate
* Kupiec likelihood ratio test
* Christoffersen independence test
* Duration-based test of conditional coverage

This component validates that the quantum-derived VaR behaves consistently with risk-management standards.

---

## **6. Summary**

The proposed methodology enables a **holistic, multi-layer comparison** between quantum and classical approaches:

* **Synthetic benchmarks** using true VaR demonstrate theoretical quantum advantage.
* **Real market backtests** evaluate practical applicability.
* **Resource analysis** establishes feasibility for near-term quantum hardware.
* **Large portfolio sampling** ensures statistically robust conclusions.

