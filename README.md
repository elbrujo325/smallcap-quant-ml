# smallcap-quant-ml

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)

> **ML-driven Small-Cap Strategy Discovery: RF → Feature Selection → Decision Tree → Rules**

---

## 📋 Flujo del Proyecto (Arquitectura ML)

| Fase | Notebook/Script | Descripción | Output | Estado |
|------|-----------------|-------------|--------|--------|
| **1. Datos + Features** | `01_eda.ipynb` | EDA + calidad datos | `data/eda_summary.csv` | ✅ |
| **2. Calibración Riesgo** | `batch_calibrate.py` | Csl + BP por activo | `data/universe_admitted.csv` | ✅ |
| **3. Feature Engineering** | `03_feature_engineering.ipynb` | ~20 indicadores OHLCV | Features en memory/DISK | 🔲 |
| **4. Triple Barrier Labeling** | `04_labeling.ipynb` | Etiquetado {0,1,2} | `data/labeled_*.csv` | 🔲 |
| **5. RF + Feature Selection** | `05_strategy_builder.ipynb` | RF → Top N features | Feature importances | 🔲 |
| **6. Decision Tree + Rules** | `05_strategy_builder.ipynb` | Tree (depth=3/4) | `data/strategies_candidates.json` | 🔲 |
| **7. Backtest + OOS** | `06_backtest_strategies.ipynb` | Train⇆Test (70/30) | `data/strategies_library.json` | 🔲 |

---

## 🛠️ Estructura de Archivos

```
smallcap-quant-ml/
├── config/
│   ├── settings.py                   # Todos los parámetros
│   └── tickers_smallcap.txt          # Lista inicial
├── scripts/
│   └── batch_calibrate.py            # Calibración Csl + BP
├── notebooks/
│   ├── 01_eda.ipynb                  # EDA puro
│   ├── 03_feature_engineering.ipynb  # ~20 indicadores
│   ├── 04_labeling.ipynb             # Triple Barrier
│   ├── 05_strategy_builder.ipynb     # RF + Tree + Reglas
│   └── 06_backtest_strategies.ipynb  # Backtest + Validación OOS
├── src/
│   ├── data.py                       # Carga datos
│   ├── features.py                   # ~20 indicadores
│   ├── risk.py                       # **BACKTESTER + Risk** (Csl, SIZE, TP/SL)
│   ├── labeling.py                   # Triple Barrier
│   ├── strategy_builder.py           # RF → Tree → Rules (NUEVO)
│   └── model.py                      # Referencia opcional
├── data/
│   ├── calibration_all.csv           # 22 tickers probados
│   ├── universe_admitted.csv         # Activos admitidos (Csl+BP ok) ✅
│   └── ... (labeled, strategies, library)
└── README.md
```

---

## ✅ Ya Implementado (Core)

### 1. **Calibración de Riesgo (Secciones 5-6)**
- `scripts/batch_calibrate.py`: Csl = MEDIANA, BP verificado independiente.
- Output: `data/universe_admitted.csv` (9 tickers admitidos).

### 2. **Backtester Genérico (`src/risk.py`)**
- `run_backtest_loop(df, entry_signal, c_sl, ...)`: Motor independiente de estrategias.
- `calculate_performance_metrics(trades, capital)`: Win Rate, PF, Sharpe, DD.
- **Se reutiliza** en Fases 10-12 para backtest de estrategias extraídas.

### 3. **Feature Engineering (`src/features.py`)**
- ~20 indicadores: RSI, ATR, EMA20/50, distancias %, VWAP, MACD, RVOL, Gap%, ADX, ROC, Momentum, Volatilidad, Bollinger, etc.

### 4. **Triple Barrier Labeling (`src/labeling.py`)**
- Labels: 0 (SL), 1 (TP), 2 (Timeout).
- Regla de desempate conservadora: SL gana si ambos se tocan simultáneamente.

### 5. **Strategy Builder (`src/strategy_builder.py`)**
- RF → Feature Importance → Top N → Decision Tree → Extract Rules.
- Filtra: P(TP) > 0.5, n_samples >= 30.

---

## ⏳ Pendiente (Por Ejecutar)

### 1. **Feature Engineering en Batch** (Fase 2)
- Calcular ~20 indicadores para todos los tickers admitidos.
- Output: `data/features_*.csv`.

### 2. **Triple Barrier Labeling** (Fase 4)
- Etiquetar entradas para cada activo.
- Output: `data/labeled_*.csv`.

### 3. **Pipeline ML Completo** (Fases 5-12)
- Ejecutar `05_strategy_builder.ipynb`: RF → Top 5 features → Tree → Reglas.
- Ejecutar `06_backtest_strategies.ipynb`: Backtest train/test + validación OOS.
- Output final: `data/strategies_library.json` (estrategias validadas).

---

## 🚀 Quick Start

**Cómo iniciar (Paso a paso, reproducible)**

- **Clonar y entrar al repositorio:** `git clone <repo-url>` y `cd smallcap-quant-ml`.
- **Crear y activar entorno virtual (recomendado):**
	- `python3 -m venv .venv`
	- `source .venv/bin/activate`
- **Instalar dependencias:**
	- Recomendado (automático): `./scripts/setup_env.sh`
	- Manual: `pip install -r requirements.txt`
	- Para reproducir exactamente el entorno del autor (opcional): `pip install -r requirements-lock.txt`
- **Verificaciones rápidas (smoke tests):**
	- `python -c "import importlib,sys; sys.path.insert(0,'.'); m=importlib.import_module('src'); print('features_v2:', hasattr(m,'add_all_features_v2'))"`
	- `python src/features.py`  # ejecuta el test incluido
	- `python src/labeling.py`  # ejecuta el test incluido
- **Generar universo admitido (calibración Csl + BP):**
	- `python scripts/batch_calibrate.py`
	- Esto genera `data/universe_admitted.csv` necesario para notebooks de backtest.
- **Abrir Jupyter / ejecutar notebooks en orden (recomendado):**
	1. `notebooks/01_eda.ipynb` — EDA y checks de calidad
	2. `notebooks/03_feature_engineering.ipynb` — calcular ~20 indicadores
	3. `notebooks/04_labeling.ipynb` — aplicar triple-barrier labeling
	4. `notebooks/05_strategy_builder.ipynb` — RF → selección → reglas
	5. `notebooks/06_backtest_strategies.ipynb` — backtest y validación OOS

	- Abrir Jupyter: `jupyter notebook` o `jupyter lab` y, antes de ejecutar, **reiniciar el kernel** y asegurarse de seleccionar el intérprete del entorno `.venv`.
	- Si no quieres reiniciar el kernel, añade al inicio de la notebook la celda:
		```python
		%load_ext autoreload
		%autoreload 2
		import importlib, sys
		sys.path.insert(0, str(Path.cwd().parent))
		import src.features
		importlib.reload(src.features)
		```

- **Comprobaciones VSCode/Jupyter:**
	- Asegúrate de que el kernel/interpretador apunta a `.venv/bin/python`.
	- Si usas Jupyter y quieres un kernel nombrado: `python -m ipykernel install --user --name=smallcap-quant-ml --display-name "smallcap-quant-ml (.venv)"`.

- **Orden mínimo recomendado de ejecución (scripts → notebooks):**
	- `./scripts/setup_env.sh` (o crear .venv e instalar deps)
	- `python scripts/batch_calibrate.py` (crea `data/universe_admitted.csv`)
	- Abrir y ejecutar `notebooks/01_eda.ipynb` (EDA)
	- Ejecutar `notebooks/03_feature_engineering.ipynb` (features batch)
	- Ejecutar `notebooks/04_labeling.ipynb` (labeling)
	- Ejecutar `notebooks/05_strategy_builder.ipynb` (RF → reglas)
	- Ejecutar `notebooks/06_backtest_strategies.ipynb` (backtest OOS)

Si algo falla durante la importación, revisa que `REPO_ROOT` en las notebooks apunte a la raíz del repo y que el kernel esté usando la `python` de `.venv`.

---

## 📐 Configuración Centralizada

Todos los parámetros en `config/settings.py`:

```python
from config.settings import DATA, FEATURES, RISK, BACKTEST

# Risk
print(RISK.risk_per_trade)      # $100
print(RISK.lookforward_window)  # 20

# Features
print(FEATURES.er_k)            # 10
print(FEATURES.atr_period)      # 50

# Backtest
print(BACKTEST.max_bars)        # 40
print(BACKTEST.tp_sl_ratio)     # 1.5
```

---

## ⚠️ Restricción de Datos

**ÚNICA fuente:** Velas OHLCV horarias (Open, High, Low, Close, Volume, timestamp).

**NADA se usa de Float, Market Cap, EV, etc.** (no existen gratis con granularidad horaria histórica).

---

## 📄 Métodología (Fases 1-12)

### Separación de Responsabilidades
- **Risk (src/risk.py)**: Determinístico, ya verificado (Csl, SIZE, TP/SL, BP).
- **ML (src/strategy_builder.py)**: Descubre patrones bajo los cuales el Risk tiene mejor probabilidad de éxito.

### Pipeline Completo
1. **Datos**: OHLCV de tickers admitidos.
2. **Features**: ~20 indicadores.
3. **Simulación**: Para cada vela, calcular SIZE, SL, TP usando `src/risk.py`.
4. **Etiquetado**: Triple Barrier (0/1/2) max 40 velas forward.
5. **Split**: 70% train (cronológico inicial), 30% test (final).
6. **Random Forest**: Sobre TODOS los ~20 features → feature_importance.
7. **Selección**: Top 3-5 features por RF.
8. **Decision Tree**: max_depth=3/4, min_samples_leaf=30.
9. **Reglas**: Cada hoja → estrategia si P(TP)>0.5 y n>=30.
10. **Backtest**: Usar `run_backtest_loop()` para cada regla (train).
11. **Validación OOS**: Backtest en 30% test → PF_test >= 0.70 * PF_train.
12. **Library**: Estrategias validadas guardadas en `data/strategies_library.json`.

---

## ⚠️ Advertencias

1. **No hay walk-forward**: Split temporal simple 70/30.
2. **No hay reinforcement desde OOS**: Las reglas se extraen solo del 70% train.
3. **Backtester reutilizado**: No se reimplementa el motor, se llama a `run_backtest_loop()`.
4. **Comisiones = 0 por ahora**: Añadir cuando tengas datos realistas.

---

<div align="center">

**By Henry Paolo Alfaro Sotil — Physicist & Data Scientist**

[![GitHub](https://img.shields.io/badge/GitHub-elbrujo325-181717?logo=github)](https://github.com/elbrujo325)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Henry%20Paolo%20Alfaro%20Sotil-0A66C2?logo=linkedin)](https://linkedin.com/in/henry-paolo-alfaro-sotil-3b75a9338)

</div>