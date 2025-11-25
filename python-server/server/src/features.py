"""
Feature engineering and risk metrics for WealthFlow project.

- Uses cleaned OHLCV data from data_loading.load_clean_data
- Computes daily returns, annual return, volatility, Sharpe ratio
- Computes correlation matrix across assets
- Generates basic EDA figures (prices + MA30, return histograms, correlation heatmap)
- Exports metrics to JSON for other modules (models_ml, portfolio)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loading import load_clean_data, TICKERS


# ---------- Paths ----------

PYTHON_ROOT = Path(__file__).resolve().parents[1]  
DATA_DIR = PYTHON_ROOT / "data"
OUTPUTS_DIR = PYTHON_ROOT / "outputs"
JSON_DIR = OUTPUTS_DIR / "json"
FIGURES_DIR = OUTPUTS_DIR / "figures"

JSON_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Core feature / risk functions ----------

def compute_returns(df: pd.DataFrame) -> pd.Series:
    """
    Compute daily returns from Close prices.

    Returns:
        pd.Series indexed by Date (if Date is index) or by integer index otherwise.
    """
    if "Close" not in df.columns:
        raise KeyError("DataFrame must contain 'Close' column")

    close = df["Close"].astype(float)
    returns = close.pct_change()
    return returns.dropna()


def compute_risk_metrics(returns: pd.Series) -> Dict[str, float]:
    """
    Compute basic risk metrics from a daily return series.

    Metrics:
        - annual_return      = mean(daily_return) * 252
        - annual_volatility  = std(daily_return) * sqrt(252)
        - sharpe             = annual_return / annual_volatility (risk-free = 0)
    """
    returns = returns.dropna().astype(float)
    if len(returns) == 0:
        return {
            "annual_return": math.nan,
            "annual_volatility": math.nan,
            "sharpe": math.nan,
        }

    mean_daily = returns.mean()
    std_daily = returns.std()

    annual_return = float(mean_daily * 252.0)
    annual_volatility = float(std_daily * np.sqrt(252.0))

    if annual_volatility == 0 or math.isnan(annual_volatility):
        sharpe = math.nan
    else:
        sharpe = float(annual_return / annual_volatility)

    return {
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
    }


def compute_moving_average(df: pd.DataFrame, window: int = 30) -> pd.Series:
    """
    Rolling mean of Close prices.

    Returns:
        pd.Series with same index as df, NaN for first (window - 1) rows.
    """
    if "Close" not in df.columns:
        raise KeyError("DataFrame must contain 'Close' column")

    return df["Close"].rolling(window=window, min_periods=1).mean()


def compute_all_assets_metrics(tickers: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    """
    For each ticker:
        - load clean data
        - compute daily returns
        - compute risk metrics (annual_return, annual_volatility, sharpe)

    Returns:
        dict[ticker] = {"annual_return": ..., "annual_volatility": ..., "sharpe": ...}
    """
    if tickers is None:
        tickers = TICKERS

    metrics: Dict[str, Dict[str, float]] = {}

    for t in tickers:
        df = load_clean_data(t)
        rets = compute_returns(df)
        metrics[t] = compute_risk_metrics(rets)

    return metrics


def compute_all_daily_returns(tickers: Optional[List[str]] = None) -> Dict[str, pd.Series]:
    """
    Compute daily returns for all tickers.

    Returns:
        dict[ticker] = pd.Series of daily returns
    """
    if tickers is None:
        tickers = TICKERS

    returns_dict: Dict[str, pd.Series] = {}

    for t in tickers:
        df = load_clean_data(t)
        rets = compute_returns(df)
        returns_dict[t] = rets

    return returns_dict


def compute_correlation_matrix(returns_dict: Dict[str, pd.Series]) -> pd.DataFrame:
    """
    Build a DataFrame of aligned daily returns and return the correlation matrix.

    Returns:
        pd.DataFrame where rows/cols are tickers, values are correlation coefficients.
    """
    # Combine all returns into a single DataFrame (align on index)
    returns_df = pd.DataFrame(returns_dict)
    # Drop rows with any NaN to have consistent sample
    returns_df = returns_df.dropna(how="any")

    corr = returns_df.corr()
    return corr


# ---------- JSON export helpers ----------

def save_asset_stats_json(
    asset_stats: Dict[str, Dict[str, float]],
    path: Optional[Path] = None,
) -> Path:
    """
    Save asset-level risk metrics to JSON.

    Schema:
        {
          "AAPL": {
            "annual_return": <float>,
            "annual_volatility": <float>,
            "sharpe": <float>
          },
          ...
        }
    """
    if path is None:
        path = JSON_DIR / "asset_stats.json"

    JSON_DIR.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to plain Python floats for JSON serialization
    serializable = {
        ticker: {
            "annual_return": float(stats.get("annual_return", math.nan)),
            "annual_volatility": float(stats.get("annual_volatility", math.nan)),
            "sharpe": float(stats.get("sharpe", math.nan)),
        }
        for ticker, stats in asset_stats.items()
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    return path


def save_correlation_matrix_json(
    corr: pd.DataFrame,
    path: Optional[Path] = None,
) -> Path:
    """
    Save correlation matrix as nested dict JSON.

    Schema:
        {
          "AAPL": { "AAPL": 1.0, "MSFT": 0.8, ... },
          "MSFT": { "AAPL": 0.8, "MSFT": 1.0, ... },
          ...
        }
    """
    if path is None:
        path = JSON_DIR / "correlation_matrix.json"

    JSON_DIR.mkdir(parents=True, exist_ok=True)

    corr_dict = corr.to_dict()

    with path.open("w", encoding="utf-8") as f:
        json.dump(corr_dict, f, indent=2)

    return path


# ---------- Plotting / EDA helpers ----------

def plot_prices_with_ma(
    tickers: Optional[List[str]] = None,
    window: int = 30,
) -> None:
    """
    Generate price + moving average plots for each ticker and save to PNG files.
    """
    if tickers is None:
        tickers = TICKERS

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for t in tickers:
        df = load_clean_data(t)
        df = df.copy()
        df["MA"] = compute_moving_average(df, window=window)

        plt.figure(figsize=(10, 5))
        plt.plot(df["Date"], df["Close"], label="Close")
        plt.plot(df["Date"], df["MA"], label=f"MA{window}")
        plt.title(f"{t} – Close Price and {window}-day MA")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()

        out_path = FIGURES_DIR / f"prices_ma{window}_{t}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()


def plot_returns_histograms(
    returns_dict: Dict[str, pd.Series],
    bins: int = 50,
) -> None:
    """
    Plot and save histogram of daily returns for each ticker.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for t, rets in returns_dict.items():
        rets = rets.dropna()

        plt.figure(figsize=(8, 4))
        plt.hist(rets, bins=bins, density=True, alpha=0.7)
        plt.title(f"{t} – Daily Returns Distribution")
        plt.xlabel("Daily Return")
        plt.ylabel("Density")
        plt.tight_layout()

        out_path = FIGURES_DIR / f"returns_hist_{t}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()


def plot_correlation_heatmap(corr: pd.DataFrame) -> None:
    """
    Plot and save correlation heatmap across assets.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    cax = plt.imshow(corr.values, interpolation="nearest", aspect="auto")
    plt.title("Asset Return Correlation Heatmap")
    plt.colorbar(cax)

    tickers = list(corr.columns)
    plt.xticks(range(len(tickers)), tickers, rotation=45, ha="right")
    plt.yticks(range(len(tickers)), tickers)

    plt.tight_layout()
    out_path = FIGURES_DIR / "correlation_heatmap.png"
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------- CLI entrypoint (for quick generation) ----------

if __name__ == "__main__":
    # Compute all returns and metrics
    returns_dict = compute_all_daily_returns()
    asset_stats = compute_all_assets_metrics()
    corr = compute_correlation_matrix(returns_dict)

    # Save JSON metrics
    save_asset_stats_json(asset_stats)
    save_correlation_matrix_json(corr)

    # Generate figures
    plot_prices_with_ma()
    plot_returns_histograms(returns_dict)
    plot_correlation_heatmap(corr)

    print("features.py: asset_stats.json, correlation_matrix.json and figures generated.")