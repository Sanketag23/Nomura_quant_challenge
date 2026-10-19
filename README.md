# Market Making Optimization

A two-part quantitative finance project built across a competitive selection process. It comprises a high-performance C++ interest rate risk engine and a Python-based ML market-making strategy, both operating on real tick data.

---

## Repository Structure

```
.
├── risk-engine/          # C++ interest rate risk engine (Part 1)
│   ├── risk_engine.cpp   # Core implementation
│   ├── Input.csv         # Calibration input: cash & swap rates, swap spec
│   └── Output.csv        # Output: discount factors, PV, par rates, DV01 risk
│
└── market-making/        # Python ML market-making strategy (Part 2)
    ├── task1_client_adversity_baseline.py   # Baseline adversity metrics per client
    ├── task2_min_half_spread.py             # Optimal minimum half-spread derivation
    ├── task3_ml_alpha_classifier.py         # ML alpha/adversity classifier
    ├── task4_externalization_threshold.py   # Client-specific externalization thresholds
    ├── task5_volatility_adaptive_quoting.py # Full market-making simulator
    ├── task*_results.csv                    # Computed output tables
    └── pnl_vs_theta.png                     # PnL vs externalization threshold plot
```

---

## Part 1: C++ Interest Rate Risk Engine

### Overview

A self-contained C++ engine that calibrates interest rate curves from raw market data and computes analytical risk sensitivities (DV01/Greeks) for multi-leg interest rate swaps using the chain rule.

### Implementation Details

**Curve Calibration & Bootstrapping**

The engine supports two interpolation schemes:
- **Linear interpolation** on discount factors between tenor nodes
- **Averaged Quadratic (AQ) interpolation** for smoother curve shapes

Curve bootstrapping proceeds sequentially across 14 tenor nodes (1W to 10Y). Short-end nodes (sub-6M) are bootstrapped directly from cash/deposit rates using:

```
DF(t) = 1 / (1 + r * DCF(0, t))
```

Longer-tenor swap nodes are solved via **Newton-Raphson root-finding**, requiring the par swap rate equation:

```
P(swap) = (p * annuity) + DF(T) - 1  =  0
```

where the annuity and final discount factor depend on previously bootstrapped nodes, making each step dependent on all prior ones.

**Analytical Greeks via Chain Rule**

Rather than finite-difference bumping, the engine computes DV01 sensitivities analytically. The chain rule decomposes sensitivity to each market input rate `c_i` as:

```
dPV/dc_i = sum_t [ (dPV/dDF_t) * (dDF_t/dDF_node) * (dDF_node/dc_i) ]
```

For swap curves, this requires propagating through the full bootstrap Jacobian `J[k][i] = dDF_node_k / dSwapRate_i`, which is computed analytically for each of the 14 nodes.

**Output**

For each of the four curve/interpolation combinations (cash-linear, cash-AQ, swap-linear, swap-AQ):
- Interpolated discount factor at a query tenor
- Present value and par rate of a specified interest rate swap
- Full 14-node DV01 risk vector (sensitivity to each market rate, in bps)

### Build

```bash
cd risk-engine
g++ -std=c++17 -O2 -o risk_engine risk_engine.cpp
./risk_engine
# Output written to Output.csv
```

---

## Part 2: Python ML Market-Making Strategy

### Overview

A full market-making pipeline applied to approximately 2 months of real equities tick data across 6 clients (A–F). The pipeline classifies trade adversity in real-time using an ML model and uses its predictions to drive a volatility-adaptive quoting strategy.

### Pipeline

#### Task 1 — Client Adversity Baseline

Computes client-level expected PnL across six forward horizons (tau = 5, 10, 15, 20, 25, 30 ticks) and classifies each client as profitable or costly to the market maker.

#### Task 2 — Minimum Half-Spread

Derives the break-even minimum half-spread `delta*` per client such that:

```
delta* = -E[PnL_m0] / E[Volume]
```

where `PnL_m0` is measured relative to the mid-price at trade time.

#### Task 3 — ML Alpha Classifier

Trains a regularized logistic regression classifier (L2 penalty, `lbfgs` solver, C tuned via grid search on the validation set) to predict whether an incoming trade is "adverse" (alpha-generating for the client, costly for the market maker) at each forward horizon.

Features include rolling volatility, time-of-day, client identity, order side, and lagged volume. The model achieves **55.87% accuracy** on the held-out test period (averaged across all six prediction horizons).

#### Task 4 — Externalization Threshold Optimization

For each client and horizon, searches for an optimal probability threshold `theta*` such that trades with predicted adversity `p > theta*` are externalized (routed away) to maximize retained PnL. Thresholds are calibrated on a validation set and applied to the held-out test set.

#### Task 5 — Volatility-Adaptive Market-Making Simulator

A full intraday market-making simulation with:
- **Volatility-adaptive spreads**: base spread scales with realized rolling volatility (bps)
- **Toxicity widening**: spread widens proportionally when predicted alpha exceeds a baseline threshold
- **Inventory skew**: bid/ask quotes are asymmetrically adjusted based on current inventory position and time-of-day urgency (eta)
- **End-of-day inventory penalty**: liquidation cost penalized quadratically

The quoting function:
```python
delta_bid = C_SIGMA * sigma * (1 + C_ALPHA * max(0, alpha - alpha_baseline)) + skew
delta_ask = C_SIGMA * sigma * (1 + C_ALPHA * max(0, alpha - alpha_baseline)) - skew
skew      = (inventory / INV_NORMALIZER) * sigma * (0.1 + C_ETA * eta^3)
```

Hyperparameters `C_SIGMA`, `C_ALPHA`, `C_ETA` are tuned on the validation set by maximizing the Sharpe score (total PnL / PnL standard deviation).

**Results on held-out test data:**
- Positive net PnL across the full simulation period
- Fill rate and daily PnL standard deviation tracked throughout
- Robust Sharpe score averaged across multiple random seeds

### Setup

```bash
pip install numpy pandas scikit-learn lightgbm matplotlib
python market-making/task5_volatility_adaptive_quoting.py
```

> Note: `trade_data.csv` is not included in this repository due to file size. Set the `TRADE_DATA_CSV` environment variable to point to your local copy before running.

---

## Results Summary

| Component | Key Metric |
|---|---|
| Risk Engine | 14-node DV01 grid, analytical chain-rule Greeks, four curve variants |
| Alpha Classifier | 55.8% test accuracy on 2-month tick data |
| Market-Making Strategy | Positive net PnL, volatility-adaptive spreads, inventory-aware skew |
