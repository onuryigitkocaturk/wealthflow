"""
ARIMA + LSTM time-series models for WealthFlow project.

This module:
- Uses cleaned OHLCV data from data_loading.load_clean_data
- Builds ARIMA models for several tickers (e.g., SPY, QQQ, TLT)
- Builds an LSTM model for at least one ticker (SPY)
- Compares ARIMA / LSTM vs baseline on a time-based train/test split
- Computes ML-based 30-day expected return and volatility
- Combines ML forecasts with baseline forecasts into outputs/json/forecasts.json

IMPORTANT:
- Does NOT modify data_loading.py, features.py, models_baseline.py
- Baseline forecasts (forecasts_baseline.json) are treated as fallback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import math

import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler

from data_loading import load_clean_data, TICKERS
from models_baseline import train_test_split_series

# ---------- Paths ----------

PYTHON_ROOT = Path(__file__).resolve().parents[1]  # .../python-server/server/ parent
OUTPUTS_DIR = PYTHON_ROOT / "outputs"
JSON_DIR = OUTPUTS_DIR / "json"

JSON_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Helper utilities ----------

def _get_price_series(ticker: str) -> pd.Series:
    """
    Load clean Close price series for a ticker, indexed by Date.
    """
    df = load_clean_data(ticker)
    if "Date" in df.columns:
        df = df.set_index("Date")
    series = df["Close"].astype(float).sort_index()
    return series


def _compute_error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute MAE and RMSE between true and predicted values.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return {"mae": math.nan, "rmse": math.nan}

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"mae": mae, "rmse": rmse}


def _prices_to_returns(prices: np.ndarray) -> np.ndarray:
    """
    Convert a price path into daily returns: r_t = p_t / p_{t-1} - 1.
    """
    prices = np.asarray(prices, dtype=float)
    if len(prices) < 2:
        return np.array([], dtype=float)

    prev = prices[:-1]
    curr = prices[1:]
    returns = curr / prev - 1.0
    return returns


# ---------- ARIMA modelling ----------

def arima_train_and_forecast(
    price_series: pd.Series,
    order: Tuple[int, int, int] = (1, 1, 1),
    split_date: str = "2023-01-01",
    horizon: int = 30,
) -> Dict[str, object]:
    """
    Fit an ARIMA model on the training part of the series, evaluate on test,
    and produce a 30-day ahead forecast from the full series.

    Returns:
        {
          "test_true": np.ndarray,
          "test_pred": np.ndarray,
          "test_dates": np.ndarray of datetime64,
          "forecast_30": np.ndarray (future prices),
          "metrics": {"mae": float, "rmse": float}
        }
    """
    series = price_series.dropna()
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Price series must have a DatetimeIndex for ARIMA.")

    # Train/test split
    train, test = train_test_split_series(series, split_date=split_date)
    train = train.dropna()
    test = test.dropna()

    if len(train) < 50 or len(test) == 0:
        # Too little data to do anything meaningful
        return {
            "test_true": np.array([]),
            "test_pred": np.array([]),
            "test_dates": np.array([]),
            "forecast_30": np.array([]),
            "metrics": {"mae": math.nan, "rmse": math.nan},
        }

    # Fit on training data and forecast test period
    model = ARIMA(train, order=order)
    fitted = model.fit()

    test_steps = len(test)
    test_pred = fitted.forecast(steps=test_steps)
    test_true = test.values
    test_dates = test.index.to_numpy()

    metrics = _compute_error_metrics(test_true, np.asarray(test_pred))

    # Fit on full series for future 30-day forecast
    model_full = ARIMA(series, order=order)
    fitted_full = model_full.fit()
    forecast_30 = fitted_full.forecast(steps=horizon)
    forecast_30 = np.asarray(forecast_30, dtype=float)

    return {
        "test_true": np.asarray(test_true, dtype=float),
        "test_pred": np.asarray(test_pred, dtype=float),
        "test_dates": test_dates,
        "forecast_30": forecast_30,
        "metrics": metrics,
    }


# ---------- LSTM modelling (for SPY) ----------

def _create_lstm_sequences(
    values_scaled: np.ndarray,
    dates: np.ndarray,
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for LSTM.

    Args:
        values_scaled: shape (n, 1) scaled price values
        dates: array-like of datetime64[ns], same length as values_scaled
        lookback: number of past days to use as input

    Returns:
        X: shape (m, lookback, 1)
        y: shape (m,)
        y_dates: shape (m,) corresponding dates of y
    """
    X_list = []
    y_list = []
    y_dates = []

    for i in range(lookback, len(values_scaled)):
        X_list.append(values_scaled[i - lookback:i, 0])
        y_list.append(values_scaled[i, 0])
        y_dates.append(dates[i])

    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=float)
    y_dates = np.array(y_dates)

    # reshape X for LSTM: (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y, y_dates


def lstm_train_and_forecast(
    price_series: pd.Series,
    split_date: str = "2023-01-01",
    lookback: int = 60,
    horizon: int = 30,
    epochs: int = 15,
    batch_size: int = 32,
) -> Dict[str, object]:
    """
    Train an LSTM model on a single price series and produce:
      - test period predictions
      - 30-day ahead price forecast

    Returns:
        {
          "test_true": np.ndarray,
          "test_pred": np.ndarray,
          "test_dates": np.ndarray of datetime64,
          "forecast_30": np.ndarray (future prices),
          "metrics": {"mae": float, "rmse": float}
        }
    """
    series = price_series.dropna()
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Price series must have a DatetimeIndex for LSTM.")

    if len(series) <= lookback + 10:
        # Not enough data
        return {
            "test_true": np.array([]),
            "test_pred": np.array([]),
            "test_dates": np.array([]),
            "forecast_30": np.array([]),
            "metrics": {"mae": math.nan, "rmse": math.nan},
        }

    df = series.to_frame(name="price").copy()
    dates = df.index.to_numpy()

    # Scale prices to [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    values = df["price"].values.reshape(-1, 1)
    values_scaled = scaler.fit_transform(values)

    # Create sequences
    X, y, y_dates = _create_lstm_sequences(values_scaled, dates, lookback=lookback)

    split_ts = pd.to_datetime(split_date)
    train_mask = y_dates <= split_ts
    test_mask = y_dates > split_ts

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        return {
            "test_true": np.array([]),
            "test_pred": np.array([]),
            "test_dates": np.array([]),
            "forecast_30": np.array([]),
            "metrics": {"mae": math.nan, "rmse": math.nan},
        }

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    test_dates = y_dates[test_mask]

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(32, input_shape=(lookback, 1)))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer=Adam(learning_rate=0.001))

    # Train
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Predict on test
    y_pred_scaled = model.predict(X_test, verbose=0).reshape(-1, 1)
    y_test_scaled = y_test.reshape(-1, 1)

    # Inverse transform to original price scale
    y_pred = scaler.inverse_transform(y_pred_scaled).ravel()
    y_true = scaler.inverse_transform(y_test_scaled).ravel()

    metrics = _compute_error_metrics(y_true, y_pred)

    # 30-day forecast: iterative prediction
    last_window_scaled = values_scaled[-lookback:, 0].copy()
    forecast_scaled = []

    for _ in range(horizon):
        inp = last_window_scaled.reshape(1, lookback, 1)
        next_scaled = model.predict(inp, verbose=0)[0, 0]
        forecast_scaled.append(next_scaled)
        # slide window
        last_window_scaled = np.roll(last_window_scaled, -1)
        last_window_scaled[-1] = next_scaled

    forecast_scaled = np.array(forecast_scaled).reshape(-1, 1)
    forecast_30 = scaler.inverse_transform(forecast_scaled).ravel()

    return {
        "test_true": y_true,
        "test_pred": y_pred,
        "test_dates": test_dates,
        "forecast_30": forecast_30,
        "metrics": metrics,
    }


# ---------- Expected 30-day return / volatility from ML forecasts ----------

def compute_expected_return_30d_ml(
    forecast_prices_30: np.ndarray,
) -> float:
    """
    Estimate expected 30-day return from a 30-day price forecast.

    Approach:
        - Compute daily returns r_t from forecast_prices_30.
        - Aggregate to 30-day horizon by compounding:
          (Π (1 + r_t)) - 1
    """
    returns = _prices_to_returns(forecast_prices_30)
    if len(returns) == 0:
        return math.nan

    gross = float(np.prod(1.0 + returns))
    exp_30d = gross - 1.0
    return exp_30d


def compute_expected_vol_30d_ml(
    forecast_prices_30: np.ndarray,
) -> float:
    """
    Estimate 30-day volatility from a 30-day price forecast.

    Here we:
        - Convert prices to daily returns.
        - Use std(returns) * sqrt(30) as a simple proxy.
    """
    returns = _prices_to_returns(forecast_prices_30)
    if len(returns) == 0:
        return math.nan

    std_daily = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    vol_30d = std_daily * math.sqrt(30.0)
    return vol_30d


# ---------- Build combined ML forecasts (ARIMA + LSTM) ----------

ARIMA_TICKERS: List[str] = ["SPY", "QQQ", "TLT"]
LSTM_TICKERS: List[str] = ["SPY"]  # only SPY for LSTM, as required


def build_ml_forecasts(
    tickers: Optional[List[str]] = None,
    split_date: str = "2023-01-01",
    horizon: int = 30,
) -> Dict[str, Dict[str, float]]:
    """
    Build ML-based 30-day forecasts for all tickers, combining:
        - ARIMA for ARIMA_TICKERS
        - LSTM for LSTM_TICKERS (currently SPY)
        - fallback to baseline forecasts otherwise

    Logic:
        - Load forecasts_baseline.json as starting point.
        - For each ticker where ARIMA/LSTM is available:
            - If both ARIMA and LSTM exist (SPY): choose the model with lower RMSE.
            - Compute expected_return_30d and expected_vol_30d from chosen forecast.
            - Override baseline values.
        - For tickers without ML models, keep baseline values.

    Returns:
        dict[ticker] = {"expected_return_30d": float, "expected_vol_30d": float}
    """
    if tickers is None:
        tickers = TICKERS

    # Load baseline forecasts
    baseline_path = JSON_DIR / "forecasts_baseline.json"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline forecasts not found: {baseline_path}")

    with baseline_path.open("r", encoding="utf-8") as f:
        baseline = json.load(f)

    final_forecasts: Dict[str, Dict[str, float]] = {}

    # Precompute ARIMA & LSTM results for tickers where we try ML
    arima_results: Dict[str, Dict[str, object]] = {}
    lstm_results: Dict[str, Dict[str, object]] = {}

    for t in tickers:
        try:
            price_series = _get_price_series(t)
        except FileNotFoundError:
            # If data missing, fallback to baseline
            continue

        # ARIMA for selected tickers
        if t in ARIMA_TICKERS:
            try:
                arima_res = arima_train_and_forecast(
                    price_series,
                    order=(1, 1, 1),
                    split_date=split_date,
                    horizon=horizon,
                )
                arima_results[t] = arima_res
            except Exception as e:
                # If ARIMA fails, leave it out (fallback to baseline)
                print(f"ARIMA failed for {t}: {e}")

        # LSTM only for selected tickers (SPY)
        if t in LSTM_TICKERS:
            try:
                lstm_res = lstm_train_and_forecast(
                    price_series,
                    split_date=split_date,
                    lookback=60,
                    horizon=horizon,
                    epochs=15,
                    batch_size=32,
                )
                lstm_results[t] = lstm_res
            except Exception as e:
                print(f"LSTM failed for {t}: {e}")

    # Build final forecasts combining ML with baseline
    for t in tickers:
        # Start from baseline, if exists
        base_entry = baseline.get(t, {})
        ml_return = None
        ml_vol = None

        # Case 1: ticker has both ARIMA and LSTM (SPY)
        if (t in arima_results) and (t in lstm_results):
            arima_rmse = arima_results[t]["metrics"]["rmse"]
            lstm_rmse = lstm_results[t]["metrics"]["rmse"]

            # Choose model with lower RMSE on test
            if math.isnan(arima_rmse) and not math.isnan(lstm_rmse):
                chosen = "lstm"
                forecast_30 = lstm_results[t]["forecast_30"]
            elif math.isnan(lstm_rmse) and not math.isnan(arima_rmse):
                chosen = "arima"
                forecast_30 = arima_results[t]["forecast_30"]
            elif not math.isnan(arima_rmse) and not math.isnan(lstm_rmse):
                if lstm_rmse <= arima_rmse:
                    chosen = "lstm"
                    forecast_30 = lstm_results[t]["forecast_30"]
                else:
                    chosen = "arima"
                    forecast_30 = arima_results[t]["forecast_30"]
            else:
                chosen = None
                forecast_30 = np.array([])

            if chosen is not None and len(forecast_30) > 0:
                ml_return = compute_expected_return_30d_ml(forecast_30)
                ml_vol = compute_expected_vol_30d_ml(forecast_30)
                print(
                    f"{t}: chosen ML model={chosen}, "
                    f"ARIMA RMSE={arima_rmse:.4f}, LSTM RMSE={lstm_rmse:.4f}, "
                    f"exp_return_30d={ml_return:.4f}, vol_30d={ml_vol:.4f}"
                )

        # Case 2: ticker only has ARIMA (e.g., QQQ, TLT)
        elif t in arima_results:
            forecast_30 = arima_results[t]["forecast_30"]
            if len(forecast_30) > 0:
                ml_return = compute_expected_return_30d_ml(forecast_30)
                ml_vol = compute_expected_vol_30d_ml(forecast_30)
                print(
                    f"{t}: using ARIMA only, "
                    f"RMSE={arima_results[t]['metrics']['rmse']:.4f}, "
                    f"exp_return_30d={ml_return:.4f}, vol_30d={ml_vol:.4f}"
                )

        # If we computed ML metrics, override baseline
        if (ml_return is not None) and (ml_vol is not None):
            final_forecasts[t] = {
                "expected_return_30d": float(ml_return),
                "expected_vol_30d": float(ml_vol),
            }
        else:
            # Fallback: keep baseline values (if any)
            if "expected_return_30d" in base_entry and "expected_vol_30d" in base_entry:
                final_forecasts[t] = {
                    "expected_return_30d": float(base_entry["expected_return_30d"])
                    if base_entry["expected_return_30d"] is not None
                    else None,
                    "expected_vol_30d": float(base_entry["expected_vol_30d"])
                    if base_entry["expected_vol_30d"] is not None
                    else None,
                }
            else:
                # If baseline also missing, just NaNs
                final_forecasts[t] = {
                    "expected_return_30d": None,
                    "expected_vol_30d": None,
                }

    return final_forecasts


def save_ml_forecasts_json(
    forecasts: Dict[str, Dict[str, float]],
    path: Optional[Path] = None,
) -> Path:
    """
    Save final ML-enhanced forecasts to JSON.

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
        path = JSON_DIR / "forecasts.json"

    JSON_DIR.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(forecasts, f, indent=2)

    return path


# ---------- CLI entrypoint ----------

if __name__ == "__main__":
    # Build ML forecasts and save to JSON
    forecasts_ml = build_ml_forecasts(
        tickers=TICKERS,
        split_date="2023-01-01",
        horizon=30,
    )
    out_path = save_ml_forecasts_json(forecasts_ml)
    print(f"models_ml.py: ML forecasts saved to {out_path}")