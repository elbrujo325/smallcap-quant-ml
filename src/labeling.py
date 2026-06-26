"""
Triple Barrier Labeling for Financial Time Series (Fase 4).

Labels:
  - 0: Stop Loss hit first (loss)
  - 1: Take Profit hit first (profit)
  - 2: Vertical barrier hit (timeout)

Rule for tie-breaker (conservative):
  If High touches TP AND Low touches SL in the SAME bar, SL wins (label=0).
  This assumes worst-case execution (slippage favors downside).

Reference: López de Prado, "Advances in Financial Machine Learning", Ch.3.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


def triple_barrier_label(df: pd.DataFrame, entry_idx: int,
                         entry_price: float, sl_price: float,
                         tp_price: float, max_bars: int = 40,
                         low_col: str = 'Low', high_col: str = 'High') -> int:
    """
    Apply Triple Barrier labeling for a single entry.
    
    Args:
        df: DataFrame with OHLC data (must have 'Low', 'High' columns)
        entry_idx: Bar index of entry
        entry_price: Entry price
        sl_price: Stop Loss price
        tp_price: Take Profit price
        max_bars: Maximum holding period (vertical barrier)
        low_col: Column name for low
        high_col: Column name for high
    
    Returns:
        Label: 0 (SL), 1 (TP), or 2 (timeout)
    
    Tie-breaker Rule (conservative):
        If High >= tp AND Low <= sl in the same bar, SL wins (label=0).
    """
    future_lows = df[low_col].iloc[entry_idx + 1: entry_idx + 1 + max_bars].values
    future_highs = df[high_col].iloc[entry_idx + 1: entry_idx + 1 + max_bars].values
    
    if len(future_lows) == 0:
        # No data forward -> timeout
        return 2
    
    n_bars = len(future_lows)
    
    # Find first SL hit
    sl_hits = np.where(future_lows <= sl_price)[0]
    tp_hits = np.where(future_highs >= tp_price)[0]
    
    # Tie-breaker: check if any bar has BOTH conditions
    for i in range(n_bars):
        if future_lows[i] <= sl_price and future_highs[i] >= tp_price:
            # SL wins (conservative assumption)
            return 0
    
    # No tie: check which happened first
    if len(sl_hits) > 0 and len(tp_hits) > 0:
        first_sl = sl_hits[0]
        first_tp = tp_hits[0]
        if first_sl <= first_tp:
            return 0  # SL first
        else:
            return 1  # TP first
    elif len(sl_hits) > 0:
        return 0  # Only SL
    elif len(tp_hits) > 0:
        return 1  # Only TP
    else:
        return 2  # Timeout


def apply_triple_barrier_to_dataset(df: pd.DataFrame, csl: float,
                                    tp_sl_ratio: float = 1.5,
                                    atr_col: str = 'ATR_50',
                                    max_bars: int = 40,
                                    entry_signal: Optional[pd.Series] = None,
                                    min_price: float = 1.0,
                                    max_price: float = 20.0) -> pd.DataFrame:
    """
    Apply Triple Barrier labeling to all valid entries in dataset.
    
    For each bar where entry is possible (or all bars if no signal provided):
    1. Calculate ATR, SL, TP
    2. Simulate forward up to max_bars
    3. Assign label (0, 1, or 2)
    
    Args:
        df: DataFrame with OHLC + ATR columns
        csl: Stop Loss ATR multiplier
        tp_sl_ratio: Take Profit / Stop Loss ratio (default 1.5)
        atr_col: Column name for ATR
        max_bars: Maximum holding period
        entry_signal: Boolean Series for entry signals. If None, label all bars.
        min_price: Minimum price for small-cap filter
        max_price: Maximum price for small-cap filter
    
    Returns:
        DataFrame with 'label' column added (only valid rows, no NaN labels)
    """
    labels = []
    valid_indices = []
    
    if entry_signal is None:
        # Label all bars (for feature importance on raw data)
        indices_to_label = df.index.tolist()
    else:
        indices_to_label = df[entry_signal].index.tolist()
    
    atr_series = df[atr_col]
    
    for idx in indices_to_label:
        # Check if we have enough forward data
        if df.index.get_loc(idx) + max_bars + 1 >= len(df):
            continue
        
        atr = atr_series.loc[idx]
        if pd.isna(atr) or atr <= 0:
            continue
        
        close = df.loc[idx, 'Close']
        if not (min_price <= close <= max_price):
            continue
        
        sl_price = close - atr * csl
        tp_price = close + atr * tp_sl_ratio * csl
        
        label = triple_barrier_label(
            df, idx, close, sl_price, tp_price, max_bars,
            low_col='Low', high_col='High'
        )
        
        labels.append(label)
        valid_indices.append(idx)
    
    result = pd.DataFrame({
        'idx': valid_indices,
        'label': labels
    })
    result = result.set_index('idx')
    result = result.sort_index()
    
    return result


def create_labeled_dataset(df: pd.DataFrame, csl: float,
                           tp_sl_ratio: float = 1.5,
                           atr_col: str = 'ATR_50',
                           max_bars: int = 40,
                           entry_signal: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Full pipeline: add labels to original DataFrame.
    
    Returns enhanced DataFrame with:
      - All original columns (OHLCV + features)
      - 'label' column (0, 1, or 2)
      - 'entry_idx' column (row index in original df)
    """
    labeled = apply_triple_barrier_to_dataset(
        df, csl, tp_sl_ratio, atr_col, max_bars, entry_signal
    )
    
    # Merge back with original data
    result = df.loc[labeled.index].copy()
    result['label'] = labeled['label'].values
    result['entry_idx'] = labeled.index
    result = result.dropna(subset=['label'])
    
    return result


if __name__ == '__main__':
    # Test simple
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    n = 100
    prices = 10 + 2 * np.cumsum(np.random.randn(n) * 0.05)
    df_test = pd.DataFrame({
        # use lower-case freq for wider pandas compatibility
        'Datetime': pd.date_range('2025-01-01', periods=n, freq='h'),
        'Open': prices,
        'High': prices * (1 + np.abs(np.random.randn(n) * 0.01)),
        'Low': prices * (1 - np.abs(np.random.randn(n) * 0.01)),
        'Close': prices,
        'Volume': np.random.randint(1000, 5000, n),
        'ATR_50': 0.2  # constant for test
    })
    
    # Simulate 3 cases
    # Case 1: TP hit
    label1 = triple_barrier_label(df_test, entry_idx=10, entry_price=10.5,
                                   sl_price=10.0, tp_price=12.0, max_bars=20)
    print(f"Case 1 (TP expected): label={label1}")
    
    # Case 2: SL hit
    label2 = triple_barrier_label(df_test, entry_idx=10, entry_price=10.5,
                                   sl_price=9.5, tp_price=12.0, max_bars=20)
    print(f"Case 2 (SL expected): label={label2}")
    
    # Case 3: Timeout
    label3 = triple_barrier_label(df_test, entry_idx=10, entry_price=10.5,
                                   sl_price=8.0, tp_price=15.0, max_bars=20)
    print(f"Case 3 (timeout expected): label={label3}")