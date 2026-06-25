"""
Risk management and position sizing utilities.
Core logic for ATR-based stop loss, take profit, and position sizing.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List

from src.features import calculate_atr as _calc_atr, filter_momentum_entries


def calculate_atr(df: pd.DataFrame, period: int = 50) -> pd.Series:
    """Calculate ATR - wrapper for consistent interface (uses features.calculate_atr)."""
    return _calc_atr(df, period=period)


def calculate_sl_tp(entry_price: float, atr: float,
                    c_sl: float, c_tp: Optional[float] = None) -> Tuple[float, float]:
    """
    Calculate Stop Loss and Take Profit prices based on ATR multipliers.

    Args:
        entry_price: Entry price
        atr: Current ATR value
        c_sl: Stop Loss multiplier (e.g., 1.9)
        c_tp: Take Profit multiplier. If None, defaults to 1.5 * c_sl
              per methodology (Ctp = 1.5 * Csl).

    Returns:
        (stop_loss_price, take_profit_price)
    """
    if c_tp is None:
        c_tp = 1.5 * c_sl  # Metodología: Ctp = 1.5 * Csl (ratio 1:1.5)

    if pd.isna(atr) or atr <= 0:
        return np.nan, np.nan

    sl_price = entry_price - atr * c_sl
    tp_price = entry_price + atr * c_tp

    return sl_price, tp_price


def calculate_position_size(atr: float, risk_per_trade: float,
                            c_sl: float) -> int:
    """
    Calculate position size based on fixed risk per trade.

    Formula: size = risk_per_trade / (c_sl * atr)

    Args:
        atr: Current ATR value
        risk_per_trade: Fixed dollar risk per trade (e.g., $100)
        c_sl: Stop Loss multiplier

    Returns:
        Integer number of shares (0 if invalid)
    """
    if pd.isna(atr) or atr <= 0:
        return 0

    sl_distance = c_sl * atr
    if sl_distance <= 0:
        return 0

    size = int(risk_per_trade / sl_distance)
    return max(0, size)


def calculate_buying_power(entry_price: float, atr: float,
                           c_sl: float, risk_per_trade: float = 100) -> float:
    """
    Calculate Buying Power required for a position.

    BP = entry_price * (risk_per_trade / (c_sl * atr))

    Args:
        entry_price: Entry price
        atr: Current ATR value
        c_sl: Stop Loss multiplier
        risk_per_trade: Fixed dollar risk per trade

    Returns:
        Buying Power in dollars
    """
    size = calculate_position_size(atr, risk_per_trade, c_sl)
    return entry_price * size


def find_optimal_coef_sl(df: pd.DataFrame, n_samples: int = 500,
                         lookforward_window: int = 20, atr_period: int = 50,
                         price_min: float = 1.0, price_max: float = 20.0,
                         momentum_k: int = 10, momentum_threshold: float = 0.3,
                         admission_range: Tuple[float, float] = (1.5, 3.0),
                         seed: int = 42) -> Optional[float]:
    """
    Calibra C_sl (metodologia.pdf Sección 5, Ecuación 14) como la MEDIANA
    de caídas implícitas en n_samples entradas aleatorias, filtradas por
    precio small-cap y momentum alcista (Efficiency Ratio). Devuelve None
    si el activo no admite (datos insuficientes o mediana fuera de rango).

    Esta función NUNCA debe buscar un valor que produzca un Buying Power
    objetivo. Esa verificación va en calculate_buying_power_distribution(),
    como paso SEPARADO Y POSTERIOR.

    Args:
        df: DataFrame OHLC completo con columnas 'High', 'Low', 'Close'
        n_samples: Número de entradas a muestrear (default 500)
        lookforward_window: Ventana forward para medir caída máxima (default 20)
        atr_period: Período para ATR (default 50)
        price_min: Precio mínimo para universo small-cap (default $1.0)
        price_max: Precio máximo para universo small-cap (default $20.0)
        momentum_k: Ventana para Efficiency Ratio (default 10)
        momentum_threshold: Umbral de ER para admitir momentum alcista (default 0.3)
        admission_range: Rango admisible para Csl mediana (default (1.5, 3.0))
        seed: Semilla aleatoria para reproducibilidad (default 42)

    Returns:
        C_sl mediana si cumple criterios de admisión, None en caso contrario

    Raises:
        None -返回 None silenciosamente cuando el activo no admite
    """
    df = df.copy()
    df['ATR'] = calculate_atr(df, period=atr_period)

    # Paso 1: filtro de universo (precio + momentum)
    valid_range = df.index[(df.index >= atr_period) &
                           (df.index <= len(df) - lookforward_window - 1)]
    price_ok = df.loc[valid_range, 'Close'].between(price_min, price_max)
    momentum_idx = set(filter_momentum_entries(df, k=momentum_k,
                                               er_threshold=momentum_threshold))
    candidates = [i for i in valid_range[price_ok.values] if i in momentum_idx]

    if len(candidates) < 30:
        return None  # datos insuficientes tras filtrar

    # Paso 2: muestreo aleatorio
    rng = np.random.RandomState(seed)
    sampled = rng.choice(candidates, size=min(n_samples, len(candidates)),
                         replace=False)

    # Paso 3: coeficiente implícito por entrada (Ecuación 14)
    coefs_sl = []
    for idx in sampled:
        atr = df.at[idx, 'ATR']
        if pd.isna(atr) or atr <= 0:
            continue
        p_entry = df.at[idx, 'Close']
        p_min = df['Low'].iloc[idx + 1: idx + 1 + lookforward_window].min()
        c_sl = (p_entry - p_min) / atr
        if c_sl > 0:
            coefs_sl.append(c_sl)

    if len(coefs_sl) < 30:
        return None

    # Paso 4: MEDIANA (no búsqueda, no promedio)
    c_sl_median = float(np.median(coefs_sl))

    # Paso 5: filtro de admisión, independiente del BP
    low, high = admission_range
    if not (low <= c_sl_median <= high):
        return None

    return c_sl_median


def calculate_buying_power_distribution(df: pd.DataFrame, c_sl: float,
                                        n_samples: int = 500,
                                        lookforward_window: int = 20,
                                        atr_period: int = 50,
                                        price_min: float = 1.0,
                                        price_max: float = 20.0,
                                        risk_per_trade: float = 100.0,
                                        seed: int = 7) -> dict:
    """
    metodologia.pdf Sección 6.2: segunda ronda INDEPENDIENTE de 500
    entradas (semilla distinta), usando el c_sl YA FIJO, únicamente para
    VERIFICAR (nunca optimizar) el Buying Power.

    Admisión (Sección 6.3): mediana(BP) en [$100,$1800] y máx(BP) ≤ $3000.

    Args:
        df: DataFrame OHLC completo con columnas 'High', 'Low', 'Close'
        c_sl: Coeficiente C_sl YA FIJO desde find_optimal_coef_sl()
        n_samples: Número de entries a muestrear (default 500)
        lookforward_window: Ventana forward (default 20)
        atr_period: Período ATR (default 50)
        price_min: Precio mínimo small-cap (default $1.0)
        price_max: Precio máximo small-cap (default $20.0)
        risk_per_trade: Riesgo por trade en dólares (default $100)
        seed: Semilla aleatoria (default 7 - distinta de find_optimal_coef_sl)

    Returns:
        Dict con:
        - median_bp: Mediana del Buying Power calculado
        - max_bp: Valor máximo del Buying Power
        - admitted: True si cumple criterios (100 ≤ median ≤ 1800 y max ≤ 3000)
        - n_valid_samples: Número de muestras válidas procesadas
    """
    df = df.copy()
    df['ATR'] = calculate_atr(df, period=atr_period)
    valid_range = df.index[(df.index >= atr_period) &
                           (df.index <= len(df) - lookforward_window - 1)]
    price_ok = df.loc[valid_range, 'Close'].between(price_min, price_max)
    candidates = valid_range[price_ok.values]

    rng = np.random.RandomState(seed)
    sampled = rng.choice(candidates, size=min(n_samples, len(candidates)),
                         replace=False)

    bp_values = []
    for idx in sampled:
        atr = df.at[idx, 'ATR']
        if pd.isna(atr) or atr <= 0:
            continue
        size = risk_per_trade / (c_sl * atr)
        bp_values.append(df.at[idx, 'Close'] * size)

    if not bp_values:
        return {'median_bp': None, 'max_bp': None, 'admitted': False,
                'n_valid_samples': 0}

    median_bp = float(np.median(bp_values))
    max_bp = float(np.max(bp_values))
    admitted = (100 <= median_bp <= 1800) and (max_bp <= 3000)
    return {'median_bp': median_bp, 'max_bp': max_bp, 'admitted': admitted,
            'n_valid_samples': len(bp_values)}


def calculate_trade_duration(df: pd.DataFrame, entry_idx: int,
                             sl_price: float, tp_price: float,
                             max_bars: int = 100) -> int:
    """
    Calculate trade duration until SL, TP, or max bars hit.

    Args:
        df: OHLC DataFrame
        entry_idx: Entry bar index
        sl_price: Stop Loss price
        tp_price: Take Profit price
        max_bars: Maximum bars to hold (Time Exit)

    Returns:
        Number of bars held
    """
    future_high = df['High'].iloc[entry_idx + 1: entry_idx + 1 + max_bars].values
    future_low = df['Low'].iloc[entry_idx + 1: entry_idx + 1 + max_bars].values

    if len(future_high) == 0:
        return 0

    hit_sl = np.where(future_low <= sl_price)[0]
    hit_tp = np.where(future_high >= tp_price)[0]

    first_sl = hit_sl[0] + 1 if len(hit_sl) > 0 else max_bars
    first_tp = hit_tp[0] + 1 if len(hit_tp) > 0 else max_bars

    return min(first_sl, first_tp, max_bars)


def calculate_durations_multiple_ratios(df: pd.DataFrame,
                                         entries_info: List[Tuple],
                                         coef_sl: float,
                                         ratios: List[float]) -> Dict[float, List[int]]:
    """
    Calculate trade durations for multiple TP/SL ratios in one pass.

    Args:
        df: OHLC DataFrame
        entries_info: List of (idx, entry_price, atr) tuples
        coef_sl: Stop Loss coefficient
        ratios: List of TP/SL ratios (e.g., [1.0, 1.5, 2.0])

    Returns:
        Dict mapping ratio -> list of durations
    """
    resultados = {ratio: [] for ratio in ratios}

    for idx, entrada, atr in entries_info:
        sl_level = entrada - coef_sl * atr
        tp_levels = {ratio: entrada + (ratio * coef_sl * atr) for ratio in ratios}

        future_high = df['High'].iloc[idx + 1:].values
        future_low = df['Low'].iloc[idx + 1:].values

        max_velas = len(future_high)
        if max_velas <= 0:
            for ratio in ratios:
                resultados[ratio].append(0)
            continue

        for ratio in ratios:
            tp_level = tp_levels[ratio]

            hit_sl = np.where(future_low <= sl_level)[0]
            hit_tp = np.where(future_high >= tp_level)[0]

            first_sl = hit_sl[0] + 1 if len(hit_sl) > 0 else max_velas
            first_tp = hit_tp[0] + 1 if len(hit_tp) > 0 else max_velas

            duracion = min(first_sl, first_tp, max_velas)
            resultados[ratio].append(duracion)

    return resultados


def run_backtest_loop(df: pd.DataFrame,
                       entry_signal: pd.Series,
                       c_sl: float = 1.9,
                       c_tp: Optional[float] = None,
                       max_bars: int = 40,
                       risk_per_trade: float = 100,
                       capital: float = 10000) -> pd.DataFrame:
    """
    Run discrete backtest loop with ATR-based risk management.

    Args:
        df: DataFrame with OHLC, ATR, and Signal columns
        entry_signal: Boolean Series for entry signals
        c_sl: Stop Loss ATR multiplier (default 1.9)
        c_tp: Take Profit ATR multiplier. If None, defaults to 1.5 * c_sl
              per methodology (Ctp = 1.5 * Csl). Explicit value allowed
              only for historical backtest comparison with explicit comment.
        max_bars: Max bars per trade (Time Exit)
        risk_per_trade: Dollar risk per trade
        capital: Starting capital

    Returns:
        DataFrame with trade results
    """
    if c_tp is None:
        c_tp = 1.5 * c_sl  # Metodología: Ctp = 1.5 * Csl (ratio 1:1.5)
    # NOTE: If explicitly passing c_tp=3.2 (ratio 1.68), this deviates from
    # documented methodology (1.5). Only do so for historical comparison.

    trades = []
    en_posicion = False

    for i in range(1, len(df)):
        if not en_posicion and entry_signal.iloc[i - 1]:
            p_entrada = df['Open'].iloc[i]
            atr_v = df['ATR'].iloc[i - 1]

            if pd.isna(atr_v) or atr_v <= 0:
                continue

            size = calculate_position_size(atr_v, risk_per_trade, c_sl)
            if size == 0:
                continue

            precio_sl, precio_tp = calculate_sl_tp(p_entrada, atr_v, c_sl, c_tp)

            en_posicion = True
            fech_in = df.index[i]
            mae_temp = p_entrada
            mfe_temp = p_entrada
            cont_velas = 0

        elif en_posicion:
            cont_velas += 1
            high_act = df['High'].iloc[i]
            low_act = df['Low'].iloc[i]
            close_act = df['Close'].iloc[i]

            mae_temp = min(mae_temp, low_act)
            mfe_temp = max(mfe_temp, high_act)

            sal_id = False
            motivo = ""
            p_salida = close_act

            if low_act <= precio_sl:
                p_salida = precio_sl
                motivo = "SL"
                sal_id = True
            elif high_act >= precio_tp:
                p_salida = precio_tp
                motivo = "TP"
                sal_id = True
            elif cont_velas >= max_bars:
                p_salida = close_act
                motivo = "Time Exit"
                sal_id = True

            if sal_id:
                pnl = (p_salida - p_entrada) * size
                trades.append({
                    'Fecha_In': fech_in,
                    'Fecha_Out': df.index[i],
                    'Precio_In': p_entrada,
                    'Precio_Out': p_salida,
                    'PnL': pnl,
                    'Motivo': motivo,
                    'MAE': p_entrada - mae_temp,
                    'MFE': mfe_temp - p_entrada,
                    'Velas': cont_velas,
                    'Size': size,
                    'SL': precio_sl,
                    'TP': precio_tp
                })
                en_posicion = False

    return pd.DataFrame(trades)


def calculate_performance_metrics(trades: pd.DataFrame,
                                   capital: float) -> Dict:
    """
    Calculate standard performance metrics from trades.

    Args:
        trades: DataFrame with trade results
        capital: Starting capital

    Returns:
        Dictionary with metrics
    """
    if len(trades) == 0:
        return {}

    wins = trades[trades['PnL'] > 0]['PnL']
    losses = trades[trades['PnL'] <= 0]['PnL']

    win_rate = len(wins) / len(trades)
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    equity = trades['PnL'].cumsum() + capital
    peak = equity.cummax()
    drawdown = (equity - peak) / peak * 100
    max_drawdown = drawdown.min()

    returns = trades['PnL'] / capital
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    pnl_total = trades['PnL'].sum()
    retorno_total = pnl_total / capital * 100

    return {
        'Total Operaciones': len(trades),
        'Win Rate': win_rate,
        'Profit Factor': profit_factor,
        'Expectancy (USD)': expectancy,
        'Avg Win': avg_win,
        'Avg Loss': avg_loss,
        'Max Drawdown (%)': max_drawdown,
        'Sharpe Ratio': sharpe,
        'PnL Total': pnl_total,
        'Retorno Total (%)': retorno_total,
        'Capital Final': capital + pnl_total
    }