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
        If High >= tp AND Low <= sl in the SAME bar as the first touch,
        SL wins (label=0). Evaluates bar-by-bar in chronological order
        and returns on first touch.
    """
    future_lows = df[low_col].iloc[entry_idx + 1: entry_idx + 1 + max_bars].values
    future_highs = df[high_col].iloc[entry_idx + 1: entry_idx + 1 + max_bars].values
    
    if len(future_lows) == 0:
        return 2
    
    for i in range(len(future_lows)):
        hit_sl = future_lows[i] <= sl_price
        hit_tp = future_highs[i] >= tp_price
        if hit_sl and hit_tp:
            return 0
        elif hit_sl:
            return 0
        elif hit_tp:
            return 1
    return 2


def apply_triple_barrier_to_dataset(df: pd.DataFrame, csl: float,
                                    tp_sl_ratio: float = 1.5,
                                    atr_col: str = 'ATR_50',
                                    max_bars: int = 40,
                                    entry_signal: Optional[pd.Series] = None,
                                    min_price: float = 1.0,
                                    max_price: float = 20.0,
                                    binary_target: bool = True) -> pd.DataFrame:
    labels = []
    valid_indices = []
    
    if entry_signal is None:
        indices_to_label = df.index.tolist()
    else:
        indices_to_label = df[entry_signal].index.tolist()
    
    atr_series = df[atr_col]
    
    for idx in indices_to_label:
        # 1. Obtener la posición entera (localización) de la vela de señal
        pos = df.index.get_loc(idx)
        
        # Guardas de seguridad para no salirte del array futuro (necesitamos pos + 1 + max_bars)
        if pos + 1 + max_bars >= len(df):
            continue
        
        atr = atr_series.iloc[pos] # ATR calculado al cierre de la señal
        if pd.isna(atr) or atr <= 0:
            continue
            
        # 2. P_entry REAL: El Open de la SIGUIENTE vela (pos + 1) tras confirmarse la señal
        p_entry = df['Open'].iloc[pos + 1]
        
        # Filtro de precio Small-Cap basado en el precio de entrada real [1, 20]
        if not (min_price <= p_entry <= max_price):
            continue
        
        # 3. Niveles de precio exactos anclados a P_entry (Ecuaciones 5 y 6 del PDF)
        sl_price = p_entry - atr * csl
        tp_price = p_entry + atr * tp_sl_ratio * csl
        
        # 4. Evaluamos las barreras desde el momento exacto en que entramos (pos + 1)
        # Modificamos el índice enviado para que 'triple_barrier_label' empiece a iterar en pos + 2
        label = triple_barrier_label(
            df, pos + 1, p_entry, sl_price, tp_price, max_bars,
            low_col='Low', high_col='High'
        )
        
        if binary_target:
            label = 1 if label == 1 else 0
        
        labels.append(label)
        valid_indices.append(idx)
    
    result = pd.DataFrame({
        'idx': valid_indices,
        'label': labels
    }).set_index('idx').sort_index()
    
    return result


def create_labeled_dataset(df: pd.DataFrame, csl: float,
                           tp_sl_ratio: float = 1.5,
                           atr_col: str = 'ATR_50',
                           max_bars: int = 40,
                           entry_signal: Optional[pd.Series] = None,
                           binary_target: bool = True) -> pd.DataFrame:
    """
    Full pipeline: add labels to original DataFrame.
    
    Returns enhanced DataFrame with:
      - All original columns (OHLCV + features)
      - 'label' column as binary TP/No-TP target by default
      - 'entry_idx' column (row index in original df)
    """
    labeled = apply_triple_barrier_to_dataset(
        df, csl, tp_sl_ratio, atr_col, max_bars, entry_signal,
        binary_target=binary_target
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