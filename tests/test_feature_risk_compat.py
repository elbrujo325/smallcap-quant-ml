import pandas as pd
from src.features import add_all_features
from src.risk import run_backtest_loop


def test_add_all_features_exposes_atr_alias_and_backtest_runs():
    df = pd.DataFrame(
        {
            "Datetime": pd.date_range("2024-01-01", periods=120, freq="h"),
            "Open": [10 + i * 0.01 for i in range(120)],
            "High": [10.5 + i * 0.01 for i in range(120)],
            "Low": [9.5 + i * 0.01 for i in range(120)],
            "Close": [10 + i * 0.01 for i in range(120)],
            "Volume": [1000] * 120,
        }
    )

    enriched = add_all_features(df)

    assert "ATR" in enriched.columns
    assert "ATR_50" in enriched.columns

    signal = pd.Series([False] * len(enriched), index=enriched.index)
    signal.iloc[60] = True

    trades = run_backtest_loop(
        df=enriched,
        entry_signal=signal,
        c_sl=1.5,
        c_tp=2.25,
        max_bars=5,
        risk_per_trade=100,
        capital=10000,
    )

    assert isinstance(trades, pd.DataFrame)
