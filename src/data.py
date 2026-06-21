"""
Data loading and preprocessing utilities.
Shared across notebooks for consistent data handling.
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import Optional, List


def load_ohlc_from_yfinance(ticker: str, period: str = "max", interval: str = "1h") -> pd.DataFrame:
    """
    Download OHLC data from Yahoo Finance.

    Args:
        ticker: Stock symbol (e.g., "ABSI")
        period: Period to download (e.g., "max", "1y", "6mo")
        interval: Data interval (e.g., "1h", "1d")

    Returns:
        DataFrame with columns: Datetime, Open, High, Low, Close, Volume
    """
    data = yf.Ticker(ticker)
    df = data.history(period=period, interval=interval)

    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}")

    df = df.reset_index()
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime').reset_index(drop=True)

    return df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]


def load_ohlc_from_csv(file_path: str, date_col: str = 'Date', time_col: str = 'Time') -> pd.DataFrame:
    """
    Load OHLC data from CSV file with separate Date/Time columns.

    Args:
        file_path: Path to CSV file
        date_col: Name of date column
        time_col: Name of time column

    Returns:
        DataFrame with Datetime index and OHLC columns
    """
    df = pd.read_csv(file_path)
    df['Datetime'] = pd.to_datetime(df[date_col] + ' ' + df[time_col])
    df = df.sort_values('Datetime').reset_index(drop=True)
    return df


def load_multiple_assets(data_dir: str, extension: str = "*.csv") -> dict:
    """
    Load multiple asset files from a directory.

    Args:
        data_dir: Directory containing asset files
        extension: File extension pattern

    Returns:
        Dict mapping asset_name -> DataFrame
    """
    assets = {}
    path = Path(data_dir)

    for file_path in path.glob(extension):
        asset_name = file_path.stem
        try:
            df = load_ohlc_from_csv(str(file_path))
            assets[asset_name] = df
        except Exception as e:
            print(f"Warning: Failed to load {asset_name}: {e}")

    return assets


def validate_ohlc_data(df: pd.DataFrame, min_rows: int = 100) -> bool:
    """Validate that DataFrame has required OHLC columns and minimum data."""
    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_cols):
        return False
    if len(df) < min_rows:
        return False
    return True


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLC data to different timeframe."""
    df = df.set_index('Datetime')
    ohlc_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    resampled = df.resample(rule).apply(ohlc_dict).dropna()
    return resampled.reset_index()