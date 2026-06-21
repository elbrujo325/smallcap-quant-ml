"""
smallcap-quant-ml: ML Pipeline for Small-Cap Trading Strategy Generation
with ATR-Adaptive Risk Management.

Modules:
- data: Data loading (yfinance, CSV, multi-asset)
- features: Technical indicators & feature engineering
- risk: ATR SL/TP, position sizing, backtest engine
- labeling: Triple barrier labeling (stubs - TODO)
- model: LightGBM, walk-forward validation (stubs - TODO)
"""

__version__ = "0.1.0"
__author__ = "Henry Paolo Alfaro Sotil"

# Public API
from src.data import (
    load_ohlc_from_yfinance,
    load_ohlc_from_csv,
    load_multiple_assets,
    validate_ohlc_data,
    resample_ohlc,
)

from src.features import (
    calculate_atr,
    calculate_sma,
    calculate_ema,
    calculate_roc,
    calculate_rsi,
    calculate_bollinger_bands,
    calculate_price_structure,
    calculate_candle_body,
    calculate_candle_range,
    calculate_upper_wick,
    calculate_lower_wick,
    add_all_features,
    generate_entry_signal,
)

from src.risk import (
    calculate_atr as _risk_calculate_atr,  # wrapper, same as features.calculate_atr
    calculate_sl_tp,
    calculate_position_size,
    calculate_buying_power,
    find_optimal_coef_sl,
    calculate_trade_duration,
    calculate_durations_multiple_ratios,
    run_backtest_loop,
    calculate_performance_metrics,
)

__all__ = [
    # data
    "load_ohlc_from_yfinance",
    "load_ohlc_from_csv",
    "load_multiple_assets",
    "validate_ohlc_data",
    "resample_ohlc",
    # features
    "calculate_atr",
    "calculate_sma",
    "calculate_ema",
    "calculate_roc",
    "calculate_rsi",
    "calculate_bollinger_bands",
    "calculate_price_structure",
    "calculate_candle_body",
    "calculate_candle_range",
    "calculate_upper_wick",
    "calculate_lower_wick",
    "add_all_features",
    "generate_entry_signal",
    # risk
    "calculate_sl_tp",
    "calculate_position_size",
    "calculate_buying_power",
    "find_optimal_coef_sl",
    "calculate_trade_duration",
    "calculate_durations_multiple_ratios",
    "run_backtest_loop",
    "calculate_performance_metrics",
]