"""
Data loading and cleaning utilities for WealthFlow project.

- Reads raw CSV files exported from Investing.com (European number format)
- Normalizes numeric columns (Price, Open, High, Low)
- Converts volume strings with M/B/K suffixes into numeric values
- Parses percentage change column into decimal returns
- Sorts by Date ascending and returns a clean DataFrame

Assumptions:
- Raw CSV files are located under:  ../data/raw/<TICKER>.csv
- Columns in raw CSV: ['Date', 'Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import math
import pandas as pd


# Adjust these paths to your own project structure if needed
PYTHON_ROOT = Path(__file__).resolve().parents[1]  # .../python/
DATA_DIR = PYTHON_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

TICKERS: List[str] = ["AAPL", "MSFT", "SPY", "QQQ", "TLT", "GLD"]


# ---------- Low-level parsing helpers ----------

def _parse_european_number(value) -> float:
    """
    Convert European formatted numeric string to float.

    Examples:
        "250,42"   -> 250.42
        "1.234,56" -> 1234.56
        "-" or ""  -> NaN
    """
    if isinstance(value, (int, float)):
        return float(value)

    if not isinstance(value, str):
        return math.nan

    value = value.strip()
    if value in ("", "-", "N/A"):
        return math.nan

    # Remove thousand separators, convert comma to dot as decimal
    value = value.replace(".", "").replace(",", ".")
    try:
        return float(value)
    except ValueError:
        return math.nan


def _parse_volume(value) -> float:
    """
    Parse volume strings like "39,48M", "1,2B", "123,45K" into numeric values.

    Examples:
        "39,48M" -> 39.48 * 1e6
        "1,2B"   -> 1.2  * 1e9
        "500K"   -> 500  * 1e3
        "-"      -> NaN
    """
    if isinstance(value, (int, float)):
        return float(value)

    if not isinstance(value, str):
        return math.nan

    value = value.strip()
    if value in ("", "-", "N/A"):
        return math.nan

    multiplier = 1.0
    last = value[-1]

    if last in ("M", "B", "K"):
        if last == "M":
            multiplier = 1e6
        elif last == "B":
            multiplier = 1e9
        elif last == "K":
            multiplier = 1e3
        value = value[:-1]  # remove suffix

    # Now parse the numeric part (European format)
    numeric_part = value.replace(".", "").replace(",", ".")
    try:
        return float(numeric_part) * multiplier
    except ValueError:
        return math.nan


def _parse_change_pct(value) -> float:
    """
    Parse change percentage strings like "-0,71%" into decimal form.

    Examples:
        "-0,71%" -> -0.0071
        "1,15%"  -> 0.0115
    """
    if isinstance(value, (int, float)):
        # Assume already in decimal, e.g. -0.0071
        return float(value)

    if not isinstance(value, str):
        return math.nan

    value = value.strip().replace("%", "")
    if value in ("", "-", "N/A"):
        return math.nan

    numeric_part = value.replace(".", "").replace(",", ".")
    try:
        return float(numeric_part) / 100.0
    except ValueError:
        return math.nan


# ---------- Core cleaning logic ----------

def clean_price_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a raw DataFrame loaded from Investing.com CSV.

    Expected columns in df:
        'Date', 'Price', 'Open', 'High', 'Low', 'Vol.', 'Change %'

    Returns a DataFrame with:
        'Date'   : datetime64[ns]
        'Close'  : float
        'Open'   : float
        'High'   : float
        'Low'    : float
        'Volume' : float
        'Change' : float (decimal, e.g. -0.0071)
    """
    df = df.copy()

    # Parse dates (dd.mm.yyyy) and sort ascending
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    # Numeric OHLC
    for col in ["Price", "Open", "High", "Low"]:
        df[col] = df[col].apply(_parse_european_number)

    # Volume
    if "Vol." in df.columns:
        df["Volume"] = df["Vol."].apply(_parse_volume)
    else:
        # Fallback if column name differs
        df["Volume"] = math.nan

    # Change %
    if "Change %" in df.columns:
        df["Change"] = df["Change %"].apply(_parse_change_pct)
    else:
        df["Change"] = math.nan

    # Drop original string columns, rename Price -> Close
    drop_cols = [c for c in ["Vol.", "Change %"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    df = df.rename(columns={"Price": "Close"})

    # Keep only the useful columns in a consistent order
    df = df[["Date", "Close", "Open", "High", "Low", "Volume", "Change"]]

    # Sort by date ascending and reset index
    df = df.sort_values("Date").reset_index(drop=True)

    return df


# ---------- Public API ----------

def load_raw_csv(
    ticker: str,
    raw_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load raw CSV for a given ticker from the raw data directory.

    Args:
        ticker: e.g. "AAPL", "MSFT", ...
        raw_dir: custom directory override (defaults to RAW_DATA_DIR)

    Returns:
        Raw DataFrame as read by pandas.read_csv
    """
    if raw_dir is None:
        raw_dir = RAW_DATA_DIR

    path = raw_dir / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Raw CSV not found for {ticker}: {path}")

    return pd.read_csv(path)


def load_clean_data(
    ticker: str,
    raw_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load and clean data for a given ticker in one step.

    This:
        - reads ../data/raw/<TICKER>.csv
        - parses dates and numbers
        - returns a clean, sorted DataFrame

    Args:
        ticker: e.g. "AAPL"
        raw_dir: optional override for raw directory

    Returns:
        Cleaned DataFrame with columns:
            Date, Close, Open, High, Low, Volume, Change
    """
    raw_df = load_raw_csv(ticker, raw_dir=raw_dir)
    clean_df = clean_price_dataframe(raw_df)
    return clean_df


def prepare_all_tickers(
    tickers: Optional[List[str]] = None,
    raw_dir: Optional[Path] = None,
    save_processed: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Load and clean all tickers, returning a dict of DataFrames.

    Args:
        tickers: list of ticker symbols; defaults to TICKERS constant.
        raw_dir: optional override for raw directory.

    Returns:
        dict mapping ticker -> cleaned DataFrame
    """
    if tickers is None:
        tickers = TICKERS

    data: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        df_clean = load_clean_data(t, raw_dir=raw_dir)
        data[t] = df_clean
        if save_processed:
            save_clean_data(t, df_clean)
    return data

def save_clean_data(
    ticker: str,
    df: pd.DataFrame,
    processed_dir: Optional[Path] = None,
) -> Path:
    """
    Save a cleaned DataFrame to the processed data directory.

    File path: <processed_dir>/<TICKER>_clean.csv
    """
    if processed_dir is None:
        processed_dir = PROCESSED_DATA_DIR

    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / f"{ticker}_clean.csv"
    df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    # Quick manual test: load all tickers and print basic info
    all_data = prepare_all_tickers()
    for ticker, df in all_data.items():
        print(f"{ticker}: {len(df)} rows, "
              f"{df['Date'].min().date()} -> {df['Date'].max().date()}")