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
├── docs/
│   └── legacy/
│       └── model.py                 # Legacy LightGBM stub
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

### 🔗 Copiar y pegar todo en terminal (para laptop fresca)

Copia y ejecuta este bloque completo en tu terminal. Al terminar, abre VSCode o tu editor favorito y ejecuta los notebooks:

```bash
# 1. Clonar el repositorio
git clone https://github.com/elbrujo325/smallcap-quant-ml.git
cd smallcap-quant-ml

# 2. Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# 3. Instalar dependencias (automático + setuptools + wheel para reproducibilidad)
./scripts/setup_env.sh

# 4. Instalar el paquete en modo editable (para desarrollo)
python3 -m pip install -e .

# 5. Verificar que todo está correcto
python -c "import src; print('✅ src version:', src.__version__)"
python src/features.py
python src/labeling.py

# 6. Generar universo admitido (calibración de riesgo)
python scripts/batch_calibrate.py

# 7. Listo para abrir en VSCode o Jupyter
# Sigue los pasos abajo para ejecutar los notebooks
```

### 📓 Después de ejecutar los comandos anteriores:

1. **Abre VSCode:**
   ```bash
   code .
   ```
   O abre tu editor de código favorito en la carpeta `smallcap-quant-ml`.

2. **Abre Jupyter Lab o Notebook:**
   ```bash
   jupyter lab
   # o
   jupyter notebook
   ```

3. **Ejecuta los notebooks en este orden:**
   - `notebooks/01_eda.ipynb` — Exploración y calidad de datos
   - `notebooks/03_feature_engineering.ipynb` — Cálculo de ~20 indicadores
   - `notebooks/04_labeling.ipynb` — Triple barrier labeling
   - `notebooks/05_strategy_builder.ipynb` — Random Forest → Feature selection → Reglas
   - `notebooks/06_backtest_strategies.ipynb` — Backtest y validación OOS

### ⚙️ Configuración de Kernel en VSCode/Jupyter:

Si Jupyter no carga el kernel correcto automáticamente, usa:

```bash
# Registrar kernel personalizado
python -m ipykernel install --user --name=smallcap-quant-ml --display-name "smallcap-quant-ml (.venv)"

# Luego en VSCode/Jupyter, selecciona este kernel antes de ejecutar las notebooks
```

---

## 📋 Explicación detallada de cada paso

**Clonar y entrar al repositorio:**
- Descarga el código fuente desde GitHub a tu laptop local

**Crear y activar entorno virtual:**
- Aísla las dependencias de este proyecto del resto del sistema
- Necesario para reproducibilidad

**Instalar dependencias:**
- `./scripts/setup_env.sh`: script automatizado que instala pip, setuptools, wheel y todos los requisitos
- Manual: `python3 -m pip install -r requirements.txt` si quieres hacerlo paso a paso
- Opcional: `python3 -m pip install -e .` para trabajar con el package en modo desarrollo

**Verificaciones rápidas (smoke tests):**
- Comprueba que todos los módulos principales carguen correctamente
- Ejecuta tests simples en `src/features.py` y `src/labeling.py`

**Generar universo admitido:**
- `python scripts/batch_calibrate.py` calibra Csl y verifica Buying Power
- Genera `data/universe_admitted.csv` que necesitan los notebooks

**Ejecutar notebooks en orden:**

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