"""
Configuración centralizada para smallcap-quant-ml.
Todos los parámetros del pipeline en un solo lugar.
"""

from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class DataConfig:
    """Configuración de datos y universo."""
    # Fuente de datos
    source: str = "yfinance"  # "yfinance" | "csv"
    interval: str = "1h"
    period: str = "730d"  # yfinance max 730d para 1h
    # Universo small cap
    price_min: float = 1.0
    price_max: float = 20.0
    min_avg_volume: int = 500_000  # volumen medio 63 barras
    max_market_cap_proxy: float = 2_000_000_000  # 2B proxy
    # Lista de tickers (se sobreescribe con archivo externo)
    tickers: List[str] = None
    tickers_file: str = "config/tickers_smallcap.txt"


@dataclass
class FeaturesConfig:
    """Configuración de features e indicadores."""
    atr_period: int = 50
    sma_period: int = 10
    roc_period: int = 5
    # Efficiency Ratio (momentum filter)
    er_k: int = 10
    er_threshold: float = 0.3
    # Estructura precio
    struct_lookback_short: int = 7
    struct_lookback_long: int = 9
    # Volumen
    vol_ma_period: int = 20
    vol_multiplier: float = 1.5


@dataclass
class RiskConfig:
    """Configuración de calibración de riesgo (Secciones 5-6 documento)."""
    # Calibración Csl
    lookforward_window: int = 20
    n_samples_csl: int = 500
    csl_admission_range: Tuple[float, float] = (1.5, 3.0)
    seed_csl: int = 42
    # Verificación Buying Power
    n_samples_bp: int = 500
    bp_median_range: Tuple[float, float] = (100.0, 1800.0)
    bp_max: float = 3000.0
    seed_bp: int = 7
    # Risk per trade fijo
    risk_per_trade: float = 100.0


@dataclass
class BacktestConfig:
    """Configuración del backtest generalizado."""
    max_bars: int = 40  # Time exit
    tp_sl_ratio: float = 1.5  # Ctp = 1.5 * Csl (metodología)
    capital: float = 10_000.0
    # Comisiones/slippage (desactivado por ahora)
    commission_per_share: float = 0.0
    slippage_pct: float = 0.0


@dataclass
class SignalConfig:
    """Configuración de reglas de entrada (parametrizables)."""
    use_er_filter: bool = True
    use_sma_trend: bool = True
    use_structure_break: bool = True
    use_volume_confirm: bool = True


# Instancias globales (importables como: from config.settings import DATA, FEATURES, ...)
DATA = DataConfig()
FEATURES = FeaturesConfig()
RISK = RiskConfig()
BACKTEST = BacktestConfig()
SIGNALS = SignalConfig()


def load_tickers_from_file(filepath: str) -> List[str]:
    """Carga lista de tickers desde archivo (uno por línea, ignora # comentarios)."""
    try:
        with open(filepath, 'r') as f:
            tickers = [line.strip().upper() for line in f if line.strip() and not line.startswith('#')]
        return tickers
    except FileNotFoundError:
        return []


def get_universe_tickers() -> List[str]:
    """Obtiene universo de tickers: archivo > config > default."""
    if DATA.tickers_file:
        tickers = load_tickers_from_file(DATA.tickers_file)
        if tickers:
            return tickers
    if DATA.tickers:
        return DATA.tickers
    # Default: algunos small caps conocidos para test
    return ["ABSI", "CRMD", "FLGT", "IMUX", "KPTI", "MCRB", "NBTX", "PRQR", "RDUS", "SGMO"]


# Validación rápida al importar
if __name__ == "__main__":
    print("=== CONFIGURACIÓN ACTUAL ===")
    print(f"Data: {DATA.source} {DATA.interval} {DATA.period}")
    print(f"Universe: ${DATA.price_min}-${DATA.price_max}, vol>{DATA.min_avg_volume:,}")
    print(f"Features: ATR({FEATURES.atr_period}), ER(k={FEATURES.er_k}, th={FEATURES.er_threshold})")
    print(f"Risk: lookforward={RISK.lookforward_window}, n_samples={RISK.n_samples_csl}")
    print(f"  Csl range: {RISK.csl_admission_range}")
    print(f"  BP median: {RISK.bp_median_range}, max: {RISK.bp_max}")
    print(f"Backtest: max_bars={BACKTEST.max_bars}, TP/SL={BACKTEST.tp_sl_ratio}:1")
    print(f"Capital: ${BACKTEST.capital:,.0f}")
    print(f"Tickers: {get_universe_tickers()[:10]}... ({len(get_universe_tickers())} total)")