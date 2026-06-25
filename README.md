# smallcap-quant-ml

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange.svg)](https://jupyter.org/)
[![Status: Active Development](https://img.shields.io/badge/Status-Active%20Development-brightgreen.svg)]()

> **ML Pipeline (LightGBM) for Systematic Small-Cap Trading Strategy Generation with ATR-Adaptive Risk Management**

---

## 📈 Overview

Unified repository consolidating three prior fragmented projects into a single, coherent end-to-end pipeline for small-cap quantitative research:

| Prior Repository | Scope | Status |
|-----------------|-------|--------|
| `small-cap-trading-bot` | Backtesting engine with ATR-based risk management | ✅ Migrated → `notebooks/01_eda.ipynb` |
| `atr-optimizacion-smallcaps` | Optimal ATR Stop Loss multipliers under Buying Power constraints | ✅ Migrated → `notebooks/02_risk_calibration.ipynb` |
| `atr-tp-analysis` | MFE/MAE analysis of Take Profit placement vs market favorable excursion | ✅ Migrated → `notebooks/03_backtest_evaluation.ipynb` |

**This repo supersedes and replaces all three.** The original repositories are archived for history.

---

## ⚙️ Tech Stack

| Category | Tools |
|----------|-------|
| **Languages** | Python 3.11+ |
| **Data & ML** | pandas, NumPy, LightGBM, scikit-learn, Optuna |
| **Financial** | yfinance, TA-Lib, custom ATR/risk modules |
| **Visualization** | Matplotlib, Seaborn |
| **Environment** | Jupyter Notebooks, pip/venv |

---

## 🔑 Key Features

- **📊 Exploratory Data Analysis** — OHLC loading, feature engineering (ATR, SMA, ROC, price structure), signal generation, discrete backtest with fixed-fractional sizing
- **🛡️ Risk Calibration** — Empirical ATR Stop Loss coefficient extraction, Buying Power-constrained optimization, multi-ratio TP/SL duration analysis
- **📈 Backtest Evaluation** — MFE/MAE analysis, TP placement quality assessment, trade filtering by exit type and time-of-day
- **🧱 Modular Source Code** — Reusable `src/` modules for data, features, risk (⚠️ partial), labeling (stub), model (stub)
- **📋 Honest Roadmap** — Clear separation of implemented vs. planned work

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/elbrujo325/smallcap-quant-ml.git
cd smallcap-quant-ml

# Create environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Optional: Install TA-Lib (required for some features)
# conda install -c conda-forge ta-lib  # recommended
# or see: https://github.com/mrjbq7/ta-lib

# Launch notebooks
jupyter lab notebooks/
```

### Run Order
1. **`notebooks/01_eda.ipynb`** — Data download, feature engineering, signal generation, discrete backtest with metrics & plots
2. **`notebooks/02_risk_calibration.ipynb`** — Load your OHLC data into `data/`, run ATR coefficient optimization under Buying Power constraints
3. **`notebooks/03_backtest_evaluation.ipynb`** — Point to your backtest results CSV, analyze MFE/MAE, TP quality, temporal patterns

---

## 📊 Methodology

**Adaptive Risk Management (ATR):** Dynamic Stop Loss and Take Profit levels scaled to asset volatility via Average True Range, with position sizing fixed at 1% capital risk per trade.

> **⚠️ Important: TP/SL Ratio** — Per documented methodology (`docs/metodologia.pdf`), `Ctp = 1.5 × Csl` (ratio 1:1.5) is the default. Legacy backtests used 1.68 for historical comparison only — see `src/risk.py` for explicit override notes.

**Efficiency Ratio Momentum Filter (Implemented):** Kaufman's Efficiency Ratio adapted for **bullish-only momentum** (signed numerator, no absolute value) — consistent with the long-only constraint (Sec. 3.2). Window `k=10` chosen as half the 20-bar lookforward window used for Csl drop measurement (Sec. 5.1.2). Entries require `ER > 0.3` to admit clean directional uptrends, discarding sideways/bearish zones.

**Triple Barrier Labeling (Planned):** López de Prado's method for sample labeling — profit target, stop loss, and vertical (time) barriers — enabling meta-labeling for primary model filtering.

**LightGBM Modeling (Planned):** Gradient boosting on engineered features with walk-forward validation, class-weighted training for imbalanced labels, and Optuna hyperparameter optimization.

**Walk-Forward Validation (Planned):** Expanding/rolling window splits respecting temporal order, preventing look-ahead bias.

> 📄 **Full mathematical detail:** See `docs/metodologia.pdf` (upload separate).

---

## ⚠️ Limitations

> **This is an academic project demonstrating application of ML to financial time series.**  
> It is **not** a validated trading system for real capital. Key limitations:
>
> - **No transaction costs** — slippage, commissions, spread not modeled
> - **In-sample backtests** — no out-of-sample validation yet (see Roadmap)
> - **Single-asset focus** — multi-asset portfolio construction not implemented
> - **Long-only** — no short-selling logic
> - **Survivorship bias** — universe selection uses current tickers only
> - **No execution simulation** — discrete backtest assumes perfect fills
> - **Csl calibration incomplete** — `find_optimal_coef_sl()` raises NotImplementedError; legacy notebook uses BP-targeted search that contradicts documented methodology — see `src/risk.py` docstring

---

## 🗺️ Roadmap

| Component | Status | Details |
|-----------|--------|---------|
| **Data Loading** | ✅ Done | `src/data.py` — yfinance, CSV, multi-asset |
| **Feature Engineering** | ✅ Done | `src/features.py` — ATR, SMA, ROC, structure, candles |
| **Risk Management** | ✅ Done | `src/risk.py` — SL/TP/sizing/backtest loop + **`find_optimal_coef_sl()` (mediana Csl) + `calculate_buying_power_distribution()` (verificación BP independiente)** implementados per metodologia.pdf Sec.5-6 |
| **EDA & Backtest Notebook** | ✅ Done | `01_eda.ipynb` — complete pipeline with metrics & plots |
| **Risk Calibration Notebook** | ✅ Done | `02_risk_calibration.ipynb` — empirical coef_SL optimization (BP-targeted fallback) |
| **Backtest Evaluation Notebook** | ✅ Done | `03_backtest_evaluation.ipynb` — MFE/MAE, TP quality |
| **Triple Barrier Labeling** | 🔲 TODO | `src/labeling.py` — stubs only |
| **LightGBM Training** | 🔲 TODO | `src/model.py` — stubs only |
| **Walk-Forward Validation** | 🔲 TODO | `src/model.py` — stubs only |
| **Optuna Hyperopt** | 🔲 TODO | Planned for model training |
| **Multi-Asset Portfolio** | 🔲 TODO | Position aggregation, correlation filters |
| **Transaction Costs** | 🔲 TODO | Realistic slippage/commission modeling |

---

## 📁 Repository Structure

```
smallcap-quant-ml/
├── README.md
├── requirements.txt
├── LICENSE
├── notebooks/
│   ├── 01_eda.ipynb                 # EDA, features, signals, backtest, metrics
│   ├── 02_risk_calibration.ipynb    # ATR coef optimization under BP constraint
│   └── 03_backtest_evaluation.ipynb # MFE/MAE analysis, TP quality assessment
├── src/
│   ├── __init__.py                  # Package exports
│   ├── data.py                      # Data loading (yfinance, CSV, multi-asset)
│   ├── features.py                  # Technical indicators & feature engineering
│   ├── risk.py                      # ATR SL/TP, position sizing, backtest engine ⚠️ partial
│   ├── labeling.py                  # Triple barrier (stubs - TODO)
│   └── model.py                     # LightGBM, walk-forward (stubs - TODO)
├── docs/
│   └── metodologia.pdf              # Upload separately
└── assets/
    └── [generated by notebooks]     # No pre-committed images
```

---

## 📸 Generated Assets

**Note:** Images are generated on-the-fly when you run the notebooks. No pre-committed assets to avoid stale/rotten links.

### 01_eda.ipynb — Strategy Results (example)

Run the notebook to generate:
- Equity curve, PnL distribution, exit reasons pie chart → saved to `../assets/resultados_estrategia.png`
- Trade-level details → `../rendimiento_detallado.csv`
- Performance metrics → `../metricas_estrategia.json`

---

## 🛠️ Development

```bash
# Run tests (when added)
pytest tests/

# Lint
ruff check src/
black src/ notebooks/

# Type check
mypy src/
```

---

## 🧪 Honest Warnings (Read Before Running)

| Issue | Location | Impact |
|-------|----------|--------|
| TP/SL ratio 1.5 vs 1.68 | `src/risk.py`, notebooks | Methodology says 1.5; legacy code uses 1.68 — explicit override required |
| No `assets/` pre-committed | Repo root | Run notebook to generate images; don't commit stale screenshots |

---

## 📄 License

MIT License — see [LICENSE](./LICENSE) for details.

---

<div align="center">

**By Henry Paolo Alfaro Sotil — Physicist & Data Scientist**

[![GitHub](https://img.shields.io/badge/GitHub-elbrujo325-181717?logo=github)](https://github.com/elbrujo325)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Henry%20Paolo%20Alfaro%20Sotil-0A66C2?logo=linkedin)](https://linkedin.com/in/henry-paolo-alfaro-sotil-3b75a9338)
[![Instagram](https://img.shields.io/badge/@lomejorphysics-E4405F?logo=instagram)](https://www.instagram.com/lomejorphysics/)

</div>