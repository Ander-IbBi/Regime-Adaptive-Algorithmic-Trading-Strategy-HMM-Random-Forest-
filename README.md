# Regime-Adaptive Trading Strategy — HMM + Random Forest

A walk-forward algorithmic trading pipeline that combines **Gaussian Hidden Markov Models** for market regime detection with **regime-specific Random Forest classifiers** for directional signal generation.

---

## Overview

The central hypothesis is that financial markets alternate between distinct statistical regimes (e.g. trending/low-volatility vs. mean-reverting/high-volatility), and that a predictive model aware of the current regime will outperform one that ignores it. Rather than fitting a single model to all market conditions, this strategy trains a dedicated classifier for each detected regime and routes predictions through the most likely active regime at each point in time.

The full pipeline is:

```
OHLCV Data  →  Feature Engineering  →  HMM Regime Detection
                                              ↓
                              Regime-Specific Random Forests
                                              ↓
                              Walk-Forward Signal Generation
                                              ↓
                              Performance Evaluation (quantstats)
```

---

## Methodology

### 1. Feature Engineering
Over 130 technical indicators are computed from OHLCV data using the `ta` library. Each indicator is tested for stationarity via the **Augmented Dickey-Fuller (ADF) test**; non-stationary series are transformed via percentage differencing before being passed to the models.

### 2. Regime Detection — Gaussian HMM
A `GaussianHMM` with K=2 hidden states is fitted on daily returns using the **Baum-Welch (EM) algorithm**. The learned transition matrix captures regime persistence. At prediction time, the current state distribution is projected one step forward using the **Chapman-Kolmogorov equation**:

$$\mathbf{p}_{t+1} = \mathbf{p}_t \cdot A$$

This gives the probability distribution over tomorrow's regime, which is used to select the appropriate classifier.

### 3. Regime-Specific Random Forests
Two `RandomForestClassifier` models are trained independently — one per HMM-labeled regime. Each predicts the binary target: whether the next-day return will be positive or negative. The signal threshold is ±0.03 around 0.50 (trades only when predicted probability > 0.53 or < 0.47).

### 4. Walk-Forward Backtest
The backtest uses a **strict walk-forward procedure** with a 4-year (≈1,008 trading days) rolling training window. At each step, all models are retrained from scratch on historical data only. A 90-day buffer between training and prediction prevents target leakage. No future information is used at any point.

---

## Project Structure

```
├── hmm_rf_trading_strategy.ipynb   # Main notebook — full pipeline with explanations
├── README.md
└── HMM_RF_Strategy_Explained.pdf   # Technical documentation (math + concepts)
```

The notebook is self-contained and walks through every step with detailed inline comments. The PDF provides a deeper mathematical treatment of the HMM, Baum-Welch algorithm, ADF test, and Random Forest, including derivations and references to the exact lines of code where each concept is used.

---

## Requirements

```bash
pip install hmmlearn ta quantstats yfinance statsmodels scikit-learn matplotlib pandas numpy
```

---

## Usage

Run the full pipeline with a single function call:

```python
results, perf = run_strategy(
    ticker       = 'SPY',
    data_start   = '2010-01-01',   # must be ≥ 4 years before signal_start
    signal_start = '2019-01-01',   # backtest starts here
    data_end     = '2024-01-01',
    num_lead     = 1               # prediction horizon in days
)
```

`ticker` accepts any Yahoo Finance symbol: US equities (`'AAPL'`), ETFs (`'SPY'`, `'GLD'`), crypto (`'BTC-USD'`), or FX (`'EURUSD=X'`).

---

## Known Limitations

This is an exploratory research implementation. Several simplifications are worth noting for anyone extending this work:

- **No transaction costs** are modelled. Frequent position changes can incur significant drag in live trading.
- The HMM observes only **univariate returns**. Richer observations (volatility, volume, VIX) would improve regime separation.
- With 130+ correlated features and limited per-regime samples, **Random Forest overfitting** is a real risk. Feature selection (e.g. mutual information ranking) is a natural next step.
- Results are sensitive to the chosen asset and backtest period. No claim is made about out-of-sample profitability.

---

