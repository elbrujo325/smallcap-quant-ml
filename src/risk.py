"""
Risk management and position sizing utilities.
Core logic for ATR-based stop loss, take profit, and position sizing.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List

from src.features import calculate_atr as _calc_atr


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


def find_optimal_coef_sl(entries_info: List[Tuple],
                         target_bp_min: float = 1200,
                         target_bp_max: float = 1500,
                         risk_per_trade: float = 100) -> Tuple[Optional[float], Optional[float]]:
    """
    Find optimal SL coefficient that keeps Buying Power in target range.

    NOT IMPLEMENTED: This function requires the full methodology from
    docs/metodologia.pdf Section 5, which specifies:
    1. Receive full OHLC DataFrame + valid random entry indices (not pre-computed ATR entries)
    2. For each entry: Δ_i = entry_price - min(Low over next 20 bars), coef_sl_implicit_i = Δ_i / ATR(50)_i
    3. MEDIAN of {coef_sl_implicit_1...500} → Csl (not mean, not BP-targeted search)
    4. Admission filter: if median ∉ [1.5, 3.0] → discard asset (return None or raise)
    5. ONLY AFTER fixing Csl median, calculate BP distribution on independent 500-entry sample
       and verify median BP ∈ [$100, $1800] and max BP ≤ $3000
    6. Csl calibration NEVER targets a BP range; BP is verified AFTER, not optimized into.

    Current implementation in notebooks/02_risk_calibration.ipynb uses a
    BP-targeted search (Phase 2) which CONTRADICTS the documented methodology.

    Args:
        entries_info: List of (idx, entry_price, atr) tuples (legacy signature)
        target_bp_min: Minimum target Buying Power (legacy, ignored)
        target_bp_max: Maximum target Buying Power (legacy, ignored)
        risk_per_trade: Risk per trade in dollars (legacy, ignored)

    Returns:
        (None, None) - function not implemented per methodology

    Raises:
        NotImplementedError: Always - see methodology document
    """
    raise NotImplementedError(
        "Csl calibration - pending correct implementation per metodologia.pdf Section 5. "
        "Current notebook logic (BP-targeted search) contradicts documented methodology. "
        "See src/risk.py docstring for required steps."
    )


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