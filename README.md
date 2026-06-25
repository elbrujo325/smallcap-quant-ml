# smallcap-quant-ml

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)

> **ML Pipeline (LightGBM) for Systematic Small-Cap Trading Strategy Generation with ATR-Adaptive Risk Management**

---

## 📋 Flujo del Proyecto (Orden Correcto)

El pipeline sigue la metodología documentada en `docs/metodologia.pdf` (Secciones 2-10):

| Fase | Notebook/Script | Output | Estado |
|------|-----------------|--------|--------|
| **1. Datos + Features + Calibración Riesgo** | `batch_calibrate.py` | `data/universe_admitted.csv` (Csl + BP) | ✅ Implementado |
| **2. EDA** | `notebooks/01_eda.ipynb` | `data/eda_summary.csv` | ✅ Implementado |
| **3. Generador de Estrategias** | `notebooks/02_strategy_generator.ipynb` | `data/strategies_generated.json` | 🔲 Pending (tuya) |
| **4. Backtest Estrategias** | `notebooks/03_backtest_strategies.ipynb` | `data/trades_backtest.csv` + metrics | 🔲 Pending |
| **5. Tests Robustez** | `notebooks/04_robustez.ipynb` | Reporte robustez | 🔲 Pending |
| **6. Validación Final** | `notebooks/05_validacion.ipynb` | Validación estadística | 🔲 Pending |
| **7. ML Pipeline** | `notebooks/05_model_training.ipynb` | Modelo LightGBM | 🔲 Pending |

---

## 🛠️ Estructura de Archivos

```
smallcap-quant-ml/
├── config/
│   ├── settings.py                   # ← Todos los parámetros centralizados
│   └── tickers_smallcap.txt          # ← Lista inicial de tickers
├── scripts/
│   └── batch_calibrate.py            # ← Calibración Csl + BP (Secciones 5-6)
├── notebooks/
│   ├── 01_eda.ipynb                  # ← EDA puro (sin backtest ni señales)
│   ├── 02_strategy_generator.ipynb   # ← GENERADOR (tú lo programas)
│   ├── 03_backtest_strategies.ipynb  # ← Backtest de estrategias generadas
│   ├── 04_robustez.ipynb             # ← 8 tests robustez (Sección 9)
│   ├── 05_validacion.ipynb           # ← 3 tests validación (Sección 10)
│   └── ... (más notebooks ML si aplica)
├── src/
│   ├── data.py                       # ← Carga datos (yfinance, CSV)
│   ├── features.py                   # ← Indicadores + Efficiency Ratio
│   ├── risk.py                       # ← **BACKTESTER GENÉRICO**: Csl, BP, run_backtest_loop()
│   ├── labeling.py                   # ← Triple Barrier (stubs)
│   └── model.py                      # ← LightGBM (stubs)
├── data/
│   ├── calibration_all.csv           # ← Todos los tickers probados
│   ├── universe_admitted.csv         # ← Atividades admitidos (Csl+BP ok) ✅
│   ├── eda_summary.csv               # ← Resumen EDA
│   └── ... (resultados de backtest/robustez)
├── docs/                             # ← Metodología PDF (pendiente subir)
└── README.md
```

---

## ✅ Ya Implementado (Herramientas Reutilizables)

### 1. **Calibración de Riesgo (Secciones 5-6)**
- `scripts/batch_calibrate.py`: Calibra Csl como MEDIANA de 500 entradas, verifica BP en muestra independiente.
- **Output**: `data/universe_admitted.csv` (activos con Csl validado y BP controlado).
- **Uso**: `python scripts/batch_calibrate.py`

### 2. **Backtester Genérico (`src/risk.py`)**
- `run_backtest_loop(df, entry_signal, c_sl, c_tp, max_bars, ...)`: Motor de backtest **independiente de la estrategia**.
- `calculate_performance_metrics(trades, capital)`: Win rate, PF, Sharpe, drawdown, etc.
- **Uso**: Importa de `src.risk` y pásale CUALQUIER señal booleana.

### 3. **Features & Indicadores (`src/features.py`)**
- ATR(50), SMA, ROC, Efficiency Ratio(k=10, th=0.3), estructura precio, velas, wicks.
- `add_all_features(df)`: Aplica todo automático.

---

## ⏳ Pendiente (Tu Turno)

### 1. **Generador de Estrategias (Crítico)**
- **Qué es**: Sistema que genera reglas de entrada/salida (motor genético, random search, GP, etc.).
- **Dónde**: `notebooks/02_strategy_generator.ipynb` (placeholder con estructura).
- **Output**: `data/strategies_generated.json` (lista de estrategias con filtros, triggers, exits).
- **Opciones**:
  - Rule-Based Builder + Random Search (recomendado: rápido, parametrizable).
  - Genético (DEAP): más complejo pero más potencia.
  - Genetic Programming: evoluciona árboles de expresión.

### 2. **Backtest de Estrategias**
- **Qué es**: Tomar estrategias generadas, convertir a señales booleanas, backtestear en `03_backtest_strategies.ipynb`.
- **Depende de**: Generador implementado.

### 3. **Tests de Robustez (Sección 9)**
- 8 tests: Noise, Variance, Monte Carlo, Reshuffle, Vs Random, Entry Lag, Walk Forward, Synthetic Data.
- **Notaback**: `04_robustez.ipynb`.

### 4. **Validación Final (Sección 10)**
- BP en estrategia real, bandas drawdown, T-Student últimos 30 trades.
- **Notebook**: `05_validacion.ipynb`.

---

## 🚀 Quick Start

```bash
# 1. Clonar
git clone https://github.com/elbrujo325/smallcap-quant-ml.git
cd smallcap-quant-ml

# 2. Ambiente
python -m venv .venv
source .venv/bin/activate

# 3. Instalar
pip install pandas numpy yfinance jupyter
# Opcional: conda install ta-lib

# 4. Calibrar Csl + BP (OPCIONAL, solo si no tienes universe_admitted.csv)
python scripts/batch_calibrate.py

# 5. EDA
jupyter notebook notebooks/01_eda.ipynb

# 6. Generar Estrategias (TODO)
jupyter notebook notebooks/02_strategy_generator.ipynb

# 7. Backtest Estrategias (TODO)
jupyter notebook notebooks/03_backtest_strategies.ipynb
```

---

## 📐 Configuración Centralizada

Todos los parámetros en `config/settings.py`:

```python
from config.settings import DATA, FEATURES, RISK, BACKTEST, SIGNALS

# Data
print(DATA.interval)        # '1h'
print(DATA.price_min)       # 1.0

# Features
print(FEATURES.atr_period)  # 50
print(FEATURES.er_k)        # 10

# Risk
print(RISK.lookforward_window)  # 20
print(RISK.csl_admission_range) # (1.5, 3.0)

# Backtest
print(BACKTEST.max_bars)        # 40
print(BACKTEST.tp_sl_ratio)     # 1.5
```

---

## ⚠️ Advertencias Importantes

1. **NO hay estrategias hardcodeadas**: El backtester es una herramienta genérica. Las estrategias deben generarse explícitamente en `02_strategy_generator.ipynb`.
2. **Csl calibrado por activo**: Cada ticker tiene su propio Csl óptimo (no usar un valor fijo).
3. **Sin comisiones**: Por ahora `BACKTEST.commission_per_share = 0.0`. Añadir cuando tengas datos realistas.
4. **Data de yfinance**: 730 días máx en 1h. Para 6 años completos necesitas CSV de TradeStation (mismo formato, ajustar en `src/data.py`).

---

## 📄 Licencia

MIT License — ver [LICENSE](./LICENSE).

---

<div align="center">

**By Henry Paolo Alfaro Sotil — Physicist & Data Scientist**

[![GitHub](https://img.shields.io/badge/GitHub-elbrujo325-181717?logo=github)](https://github.com/elbrujo325)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Henry%20Paolo%20Alfaro%20Sotil-0A66C2?logo=linkedin)](https://linkedin.com/in/henry-paolo-alfaro-sotil-3b75a9338)

</div>