"""
Technical indicator calculations and feature engineering.
Shared across notebooks for consistent indicator computation.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple


def calculate_atr(df: pd.DataFrame, period: int = 14, high_col: str = 'High',
                  low_col: str = 'Low', close_col: str = 'Close') -> pd.Series:
    """
    Calculate Average True Range (ATR).

    Args:
        df: DataFrame with OHLC columns
        period: ATR period (default 14)
        high_col: High column name
        low_col: Low column name
        close_col: Close column name

    Returns:
        Series with ATR values
    """
    high = df[high_col]
    low = df[low_col]
    prev_close = df[close_col].shift(1)

    tr = pd.concat([
        high - low,
        abs(high - prev_close),
        abs(low - prev_close)
    ], axis=1).max(axis=1)

    return tr.rolling(window=period).mean()


def calculate_sma(df: pd.DataFrame, period: int, column: str = 'Close') -> pd.Series:
    """Simple Moving Average."""
    return df[column].rolling(window=period).mean()


def calculate_ema(df: pd.DataFrame, period: int, column: str = 'Close') -> pd.Series:
    """Exponential Moving Average."""
    return df[column].ewm(span=period, adjust=False).mean()


def calculate_roc(df: pd.DataFrame, period: int, column: str = 'Close') -> pd.Series:
    """Rate of Change (momentum)."""
    return (df[column] / df[column].shift(period) - 1) * 100


def calculate_rsi(df: pd.DataFrame, period: int = 14, column: str = 'Close') -> pd.Series:
    """Relative Strength Index."""
    delta = df[column].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20,
                              num_std: float = 2, column: str = 'Close') -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands: (upper, middle, lower)."""
    middle = df[column].rolling(window=period).mean()
    std = df[column].rolling(window=period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def calculate_price_structure(df: pd.DataFrame, lookback_short: int = 7,
                               lookback_long: int = 9) -> pd.Series:
    """
    Price structure condition: Open[t-lookback_short] > High[t-lookback_long]
    Indicates breakout from previous range.
    """
    return df['Open'].shift(lookback_short) > df['High'].shift(lookback_long)


def calculate_candle_body(df: pd.DataFrame) -> pd.Series:
    """Candle body size (abs(Close - Open))."""
    return abs(df['Close'] - df['Open'])


def calculate_candle_range(df: pd.DataFrame) -> pd.Series:
    """Full candle range (High - Low)."""
    return df['High'] - df['Low']


def calculate_upper_wick(df: pd.DataFrame) -> pd.Series:
    """Upper wick size."""
    return df['High'] - df[['Open', 'Close']].max(axis=1)


def calculate_lower_wick(df: pd.DataFrame) -> pd.Series:
    """Lower wick size."""
    return df[['Open', 'Close']].min(axis=1) - df['Low']


def add_all_features(df: pd.DataFrame, atr_period: int = 50,
                     sma_period: int = 10, roc_period: int = 5) -> pd.DataFrame:
    """
    Add all standard features to DataFrame.

    Args:
        df: DataFrame with OHLC columns
        atr_period: ATR period
        sma_period: SMA period
        roc_period: ROC period

    Returns:
        DataFrame with added feature columns
    """
    df = df.copy()

    df['ATR'] = calculate_atr(df, period=atr_period)
    df['SMA'] = calculate_sma(df, period=sma_period)
    df['ROC'] = calculate_roc(df, period=roc_period)
    df['Estructura_OK'] = calculate_price_structure(df)
    df['Candle_Body'] = calculate_candle_body(df)
    df['Candle_Range'] = calculate_candle_range(df)
    df['Upper_Wick'] = calculate_upper_wick(df)
    df['Lower_Wick'] = calculate_lower_wick(df)

    return df


def generate_entry_signal(df: pd.DataFrame,
                          price_min: float = 1.0,
                          price_max: float = 20.0) -> pd.Series:
    """
    Generate entry signal based on standard criteria.

    Conditions:
    - Price in small-cap range
    - Price above SMA (trend)
    - ROC decaying (momentum fading)
    - Favorable price structure
    """
    return (
        (df['Close'] > price_min) & (df['Close'] <= price_max) &
        (df['Close'] > df['SMA']) &
        (df['ROC'] < df['ROC'].shift(3)) &
        (df['Estructura_OK'] == True)
    )