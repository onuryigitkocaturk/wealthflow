"""
Baseline time-series models for WealthFlow project.

This module:
- Implements naive and moving-average forecast baselines.
- Provides helper functions for time-based train/test splits.
- Computes simple 30-day expected return and volatility from historical data.
- Exports baseline forecasts to JSON (forecasts_baseline.json), which can be
  used directly or as a fallback by more advanced models (ARIMA/LSTM).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import math

import numpy as np
import pandas as pd

from data_loading import load_clean_data, TICKERS


# ---------- Paths ----------

PYTHON_ROOT = Path(__file__).resolve().parents[1]  # .../codes/python/
OUTPUTS_DIR = PYTHON_ROOT / "outputs"
JSON_DIR = OUTPUTS_DIR / "json"

JSON_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Train/test split helper ----------

def train_test_split_series(series: pd.Series, split_date: str) -> Tuple[pd.Series, pd.Series]:
    """
    Time-based split of a price series.

    Args:
        series: price series with a Date index or a column convertible to datetime.
        split_date: e.g. "2023-01-01" – all observations <= split_date -> train, else -> test.

    Returns:
        (train_series, test_series)
    """
    # If series has a DatetimeIndex, we can directly mask it
    if isinstance(series.index, pd.DatetimeIndex):
        split_ts = pd.to_datetime(split_date)
        train = series[series.index <= split_ts]
        test = series[series.index > split_ts]
    else:
        # Fallback: assume we have a corresponding Date column somewhere else.
        # In most of our use-cases we will convert the index to DatetimeIndex before calling this.
        raise ValueError("Series must have a DatetimeIndex for time-based split.")

    return train, test


# ---------- Baseline forecasting models ----------

def naive_forecast_last_value(series: pd.Series, horizon: int = 30) -> np.ndarray:
    """
    Naive baseline: use the last observed value as forecast for all future steps.

    Args:
        series: history of prices
        horizon: number of future steps to forecast

    Returns:
        np.ndarray of shape (horizon,) with constant value = last observed price
    """
    series = series.dropna()
    if len(series) == 0:
        return np.full(horizon, np.nan, dtype=float)

    last_value = float(series.iloc[-1])
    return np.full(horizon, last_value, dtype=float)


def moving_average_forecast(
    series: pd.Series,
    window: int = 30,
    horizon: int = 30,
) -> np.ndarray:
    """
    Moving-average baseline: use the mean of the last `window` observations
    as a flat forecast for the next `horizon` steps.

    Args:
        series: history of prices
        window: number of last observations to average
        horizon: forecast horizon

    Returns:
        np.ndarray of shape (horizon,) with the moving-average value
    """
    series = series.dropna()
    if len(series) == 0:
        return np.full(horizon, np.nan, dtype=float)

    if len(series) < window:
        window_series = series
    else:
        window_series = series.iloc[-window:]

    ma_value = float(window_series.mean())
    return np.full(horizon, ma_value, dtype=float)


# ---------- Expected 30-day return / volatility from history ----------

def _compute_recent_daily_returns(price_series: pd.Series, lookback_days: int = 60) -> pd.Series:
    """
    Helper: compute daily returns from price series and keep only the last N days.
    """
    returns = price_series.pct_change().dropna()

    if len(returns) <= lookback_days:
        return returns
    return returns.iloc[-lookback_days:]


def compute_expected_return_30d_from_history(
    price_series: pd.Series,
    lookback_days: int = 60,
) -> float:
    """
    Estimate expected 30-day return based on recent daily returns.

    Approach:
        - Compute daily returns from prices.
        - Take the last `lookback_days` of daily returns.
        - Expected daily return = mean of recent daily returns.
        - Expected 30-day return ≈ expected_daily_return * 30.

    Returns:
        float (decimal), e.g. 0.03 for +3% expected 30-day return.
    """
    recent_returns = _compute_recent_daily_returns(price_series, lookback_days=lookback_days)
    if len(recent_returns) == 0:
        return math.nan

    mean_daily = float(recent_returns.mean())
    expected_30d_return = mean_daily * 30.0
    return expected_30d_return


def compute_expected_vol_30d_from_history(
    price_series: pd.Series,
    lookback_days: int = 60,
) -> float:
    """
    Estimate expected 30-day volatility based on recent daily returns.

    Approach:
        - Compute daily returns from prices.
        - Take the last `lookback_days` of daily returns.
        - Daily volatility = std of recent daily returns.
        - 30-day volatility ≈ daily_vol * sqrt(30).

    Returns:
        float, approximate 30-day volatility (standard deviation of 30-day return).
    """
    recent_returns = _compute_recent_daily_returns(price_series, lookback_days=lookback_days)
    if len(recent_returns) == 0:
        return math.nan

    std_daily = float(recent_returns.std())
    vol_30d = std_daily * math.sqrt(30.0)
    return vol_30d


# ---------- Baseline forecasts JSON ----------

def build_baseline_forecasts(
    tickers: Optional[List[str]] = None,
    lookback_days: int = 60,
) -> Dict[str, Dict[str, float]]:
    """
    Build baseline 30-day expected return and volatility for all tickers.

    For each ticker:
        - Load clean Close price series.
        - Compute expected 30-day return from recent history.
        - Compute expected 30-day volatility from recent history.

    Returns:
        dict[ticker] = {"expected_return_30d": float, "expected_vol_30d": float}
    """
    if tickers is None:
        tickers = TICKERS

    forecasts: Dict[str, Dict[str, float]] = {}

    for t in tickers:
        df = load_clean_data(t)
        # Ensure Date is index for potential time-based operations
        if "Date" in df.columns:
            df = df.set_index("Date")

        price_series = df["Close"].astype(float)

        exp_ret_30d = compute_expected_return_30d_from_history(
            price_series, lookback_days=lookback_days
        )
        exp_vol_30d = compute_expected_vol_30d_from_history(
            price_series, lookback_days=lookback_days
        )

        forecasts[t] = {
            "expected_return_30d": float(exp_ret_30d) if not math.isnan(exp_ret_30d) else None,
            "expected_vol_30d": float(exp_vol_30d) if not math.isnan(exp_vol_30d) else None,
        }

    return forecasts


def save_baseline_forecasts_json(
    forecasts: Dict[str, Dict[str, float]],
    path: Optional[Path] = None,
) -> Path:
    """
    Save baseline forecasts to JSON.

    Schema:
        {
          "AAPL": {
            "expected_return_30d": <float>,
            "expected_vol_30d": <float>
          },
          ...
        }
    """
    if path is None:
        path = JSON_DIR / "forecasts_baseline.json"

    JSON_DIR.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(forecasts, f, indent=2)

    return path


# ---------- CLI entrypoint ----------

if __name__ == "__main__":
    forecasts = build_baseline_forecasts()
    out_path = save_baseline_forecasts_json(forecasts)
    print(f"models_baseline.py: baseline forecasts saved to {out_path}")