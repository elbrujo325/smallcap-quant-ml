"""
smallcap-quant-ml: ML Pipeline for Small-Cap Trading Strategy Generation
with ATR-Adaptive Risk Management.

Modules:
- data: Data loading (yfinance, CSV, multi-asset)
- features: Technical indicators & feature engineering
- risk: ATR SL/TP, position sizing, backtest engine
- labeling: Triple barrier labeling
- strategy_builder: RF → Feature Selection → Decision Tree → Rules
- model: LightGBM, walk-forward validation (stubs - TODO)
"""

__version__ = "0.2.0"
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
    calculate_rsi,
    calculate_macd,
    calculate_vwap,
    calculate_adx,
    calculate_momentum,
    calculate_volatility,
    calculate_gap,
    calculate_rvol,
    calculate_efficiency_ratio,
    calculate_bollinger_dist,
    add_all_features_v2,
    filter_momentum_entries,
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

from src.labeling import (
    triple_barrier_label,
    apply_triple_barrier_to_dataset,
    create_labeled_dataset,
)

from src.strategy_builder import (
    StrategyBuilder,
    StrategyRule,
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
    "calculate_rsi",
    "calculate_macd",
    "calculate_vwap",
    "calculate_adx",
    "calculate_momentum",
    "calculate_volatility",
    "calculate_gap",
    "calculate_rvol",
    "calculate_efficiency_ratio",
    "calculate_bollinger_dist",
    "add_all_features_v2",
    "filter_momentum_entries",
    # risk
    "calculate_sl_tp",
    "calculate_position_size",
    "calculate_buying_power",
    "find_optimal_coef_sl",
    "calculate_trade_duration",
    "calculate_durations_multiple_ratios",
    "run_backtest_loop",
    "calculate_performance_metrics",
    # labeling
    "triple_barrier_label",
    "apply_triple_barrier_to_dataset",
    "create_labeled_dataset",
    # strategy_builder
    "StrategyBuilder",
    "StrategyRule",
]