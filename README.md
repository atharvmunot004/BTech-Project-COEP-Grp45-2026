# BTech-Project-COEP-Grp45-2026
This repo contains our BTech Project. We plan on building a Quantum Hedge Fund


quant-fund/
  services/
    api/                # FastAPI: health, positions, orders (read-only in MVP)
    ingest/             # market data pullers + websocket consumers
    executor/           # order routing, risk checks, broker adapters
    research/           # notebooks, backtests, ML/RL training
    risk/               # VaR/CVaR/GARCH/EVT jobs
    optimizer/          # mean-variance, BL, CVaR, QMV wrappers
    regime/             # HMM/GMM/RS-GARCH, qPCA/QBM pipelines
    quantum/            # QAE, QAOA, QGAN modules (Qiskit/PennyLane)
  libs/
    core/               # common: data schemas, feature builders, slippage models
    brokers/            # kiteconnect.py, ib.py, alpaca.py, ccxt.py
    storage/            # postgres/timescale clients, MinIO, MLflow utils
  infra/
    docker-compose.yaml
    k8s/                # manifests for k3s (later)
    grafana/            # dashboards json
  tests/
  .env.example
  README.md



```python
class FundAgent:
    def __init__(self, tools, store, policy):
        self.t = tools         # dict of callables
        self.db = store        # positions, prices, runs
        self.policy = policy   # risk caps, routes

    def run_daily(self):
        prices = self.t["get_prices"](symbols=self.policy.symbols, interval="1d", lookback=365*3)
        feats   = self.t["calc_features"](prices)
        regime  = self.t["detect_regime"](feats, method=self.policy.regime_method)
        signals = self._choose_signals(regime, feats)
        mu, Sigma = self._estimate_mu_sigma(signals, prices)

        alloc = self._optimize(mu, Sigma)                 # opt_markowitz / opt_cvar / opt_qmv
        risk  = self.t["risk_cvar"](alloc, prices, alpha=0.95)

        if self._violates_caps(risk):                     # hard guardrails
            alloc = self._scale_down(alloc, risk)

        orders = self._diff_to_orders(alloc, self.t["current_positions"]())
        self.db.log_run(regime, alloc, risk, orders)

        if self.policy.mode in ("paper","live") and orders:
            self.t["place_orders"](orders, mode=self.policy.mode)

        return {"regime": regime, "alloc": alloc, "risk": risk, "orders": orders}

    # ... helper methods (choose_signals, optimize, scale_down, diff_to_orders)

```