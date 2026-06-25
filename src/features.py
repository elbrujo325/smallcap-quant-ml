"""
Technical indicator calculations and feature engineering.
Expanded feature set for ML-driven strategy discovery (~20 indicators).
All OHLCV-based, no external data (Float, Market Cap, etc.).
"""

import pandas as pd
import numpy as np
from typing Optional, Tuple


def calculate_atr(df: pd.DataFrame, period: int = 14, high_col: str = 'High',
                  low_col: str = 'Low', close_col: str = 'Close') -> pd.Series:
    """Calculate Average True Range (ATR)."""
    high = df[high_col]
    low = df[low_col]
    prev_close = df[close_col].shift(1)
    tr = pd.concat([high - low, abs(high - prev_close), abs(low - prev_close)], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calculate_sma(df: pd.DataFrame, period: int, column: str = 'Close') -> pd.Series:
    """Simple Moving Average."""
    return df[column].rolling(window=period).mean()


def calculate_ema(df: pd.DataFrame, period: int, column: str = 'Close') -> pd.Series:
    """Exponential Moving Average."""
    return df[column].ewm(span=period, adjust=False).mean()


def calculate_rsi(df: pd.DataFrame, period: int = 14, column: str = 'Close') -> pd.Series:
    """Relative Strength Index (RSI)."""
    delta = df[column].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9,
                   column: str = 'Close') -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD: (macd_line, signal_line, histogram)."""
    ema_fast = df[column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_vwap(df: pd.DataFrame, window: int = 20,
                   high_col: str = 'High', low_col: str = 'Low',
                   close_col: str = 'Close', vol_col: str = 'Volume') -> pd.Series:
    """Rolling VWAP (no daily reset, window-based)."""
    typical_price = (df[high_col] + df[low_col] + df[close_col]) / 3
    cum_vol = df[vol_col].rolling(window=window).sum()
    cum_tp_vol = (typical_price * df[vol_col]).rolling(window=window).sum()
    vwap = cum_tp_vol / cum_vol
    return vwap


def calculate_adx(df: pd.DataFrame, period: int = 14, high_col: str = 'High',
                  low_col: str = 'Low', close_col: str = 'Close') -> pd.Series:
    """Average Directional Index (ADX)."""
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    
    smoothed_plus = plus_dm.ewm(span=period, adjust=False).mean()
    smoothed_minus = minus_dm.ewm(span=period, adjust=False).mean()
    adx = 100 * (abs(smoothed_plus - smoothed_minus) / (smoothed_plus + smoothed_minus)
                 .replace(0, np.nan)).ewm(span=period, adjust=False).mean()
    return adx


def calculate_momentum(df: pd.DataFrame, period: int = 10, column: str = 'Close') -> pd.Series:
    """Rate of Change / Momentum."""
    return df[column] / df[column].shift(period) - 1


def calculate_volatility(df: pd.DataFrame, period: int = 20, column: str = 'Close') -> pd.Series:
    """Rolling volatility (std of returns)."""
    returns = df[column].pct_change()
    return returns.rolling(window=period).std() * np.sqrt(252)


def calculate_gap(df: pd.DataFrame, close_col: str = 'Close', open_col: str = 'Open') -> pd.Series:
    """Gap %: (Open_t - Close_{t-1}) / Close_{t-1}."""
    prev_close = df[close_col].shift(1)
    return (df[open_col] - prev_close) / prev_close


def calculate_rvol(df: pd.DataFrame, period: int = 20, vol_col: str = 'Volume') -> pd.Series:
    """Relative Volume: Volume / SMA(Volume)."""
    vol_ma = df[vol_col].rolling(window=period).mean()
    return df[vol_col] / vol_ma


def calculate_efficiency_ratio(df: pd.DataFrame, k: int = 10, column: str = 'Close') -> pd.Series:
    """Kaufman Efficiency Ratio (signed numerator, long-only)."""
    price = df[column]
    net_change = price - price.shift(k)
    path_length = price.diff().abs().rolling(window=k).sum()
    er = net_change / path_length
    return er


def calculate_bollinger_dist(df: pd.DataFrame, period: int = 20, num_std: float = 2,
                             column: str = 'Close') -> Tuple[pd.Series, pd.Series]:
    """Distance to Bollinger Bands (% from mid)."""
    middle = df[column].rolling(window=period).mean()
    std = df[column].rolling(window=period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    dist_upper = (df[column] - upper) / middle * 100
    dist_lower = (df[column] - lower) / middle * 100
    return dist_upper, dist_lower


def add_all_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ALL ~20 indicators for ML feature engineering (Fase 2).
    Returns DataFrame with original OHLCV + all computed features.
    """
    df = df.copy()
    
    # 1. RSI(14)
    df['RSI_14'] = calculate_rsi(df, period=14)
    
    # 2. ATR(50) - already exists, standardize
    df['ATR_50'] = calculate_atr(df, period=50)
    
    # 3-6. EMA20, EMA50 + distancias %
    df['EMA_20'] = calculate_ema(df, period=20)
    df['EMA_50'] = calculate_ema(df, period=50)
    df['dist_ema20_pct'] = (df['Close'] - df['EMA_20']) / df[' EMA_20'] * 100
    df['dist_ema50_pct'] = (df['Close'] - df['EMA_50']) / df['EMA_50'] * 100
    
    # 7-8. VWAP rodante + distancia %
    df['VWAP_20'] = calculate_vwap(df, window=20)
    df['dist_vwap_pct'] = (df['Close'] - df['VWAP_20']) / df['VWAP_20'] * 100
    
    # 9-11. MACD
    df['MACD_line'], df['MACD_signal'], df['MACD_hist'] = calculate_macd(df)
    
    # 12. RVOL
    df['RVOL_20'] = calculate_rvol(df, period=20)
    
    # 13. Efficiency Ratio
    df['ER_Kaufman_10'] = calculate_efficiency_ratio(df, k=10)
    
    # 14. Gap %
    df['Gap_pct'] = calculate_gap(df)
    
    # 15. ADX
    df['ADX_14'] = calculate_adx(df, period=14)
    
    # 16. ROC
    df['ROC_10'] = calculate_momentum(df, period=10) * 100
    
    # 17. Momentum
    df['Momentum_10'] = calculate_momentum(df, period=10)
    
    # 18. Volatilidad
    df['Volatility_20'] = calculate_volatility(df, period=20)
    
    # 19. SMA(10)
    df['SMA_10'] = calculate_sma(df, period=10)
    
    # 20. Bollinger Bands distancias
    df['dist_bb_upper_pct'], df['dist_bb_lower_pct'] = calculate_bollinger_dist(df)
    
    # Candle patterns (legacy)
    df['Candle_Body'] = abs(df['Close'] - df['Open'])
    df['Candle_Range'] = df['High'] - df['Low']
    df['Upper_Wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['Lower_Wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    
    # Fill NaNs (for early rows)
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df


def filter_momentum_entries(df: pd.DataFrame, k: int = 10,
                            er_threshold: float = 0.3,
                            column: str = 'Close') -> np.ndarray:
    """Filtro de momentum: ER > threshold."""
    er = calculate_efficiency_ratio(df, k=k, column=column)
    valid = er > er_threshold
    return df.index[valid.fillna(False)].to_numpy()


if __name__ == '__main__':
    # Test rápido
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    n = 1000
    prices = 5 + 3 * np.cumsum(np.random.randn(n) * 0.02)
    df_test = pd.DataFrame({
        'Datetime': pd.date_range('2025-01-01', periods=n, freq='H'),
        'Open': prices * (1 + np.random.randn(n) * 0.001),
        'High': prices * (1 + np.abs(np.random.randn(n)) * 0.002),
        'Low': prices * (1 - np.abs(np.random.randn(n)) * 0.002),
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, n)
    })
    
    df_test = add_all_features_v2(df_test)
    print("Features creadas:")
    print([c for c in df_test.columns if c not in ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']])