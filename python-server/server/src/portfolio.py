"""
Portfolio construction and robo-advisor logic for WealthFlow.

Uses:
- outputs/json/asset_stats.json
- outputs/json/correlation_matrix.json
- outputs/json/forecasts.json (ML + baseline combined)

Responsibilities:
- Define static base weights for risk profiles (conservative, balanced, aggressive)
- Load asset-level stats, correlations, and 30-day forecasts
- Compute portfolio expected return and risk (sqrt(w^T Σ w))
- Adjust base weights slightly based on forecasted 30-day returns
- Export final recommendations to outputs/json/portfolio_recommendations.json

This module does NOT touch data_loading.py, features.py, models_baseline.py, models_ml.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import json
import math

import numpy as np
import pandas as pd

from data_loading import TICKERS

# ---------- Paths ----------

PYTHON_ROOT = Path(__file__).resolve().parents[1]  # .../server/
OUTPUTS_DIR = PYTHON_ROOT / "outputs"
JSON_DIR = OUTPUTS_DIR / "json"

JSON_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Base weights for risk profiles ----------

# Weights sum to 1.0 in each profile.
# Conservative: high TLT (bonds) and GLD (gold), lower equities.
# Balanced: mix of equities + some bonds/gold.
# Aggressive: overweight growth/equities (QQQ, AAPL, MSFT, SPY), low TLT/GLD.

BASE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "conservative": {
        "AAPL": 0.10,
        "MSFT": 0.10,
        "SPY":  0.20,
        "QQQ":  0.10,
        "TLT":  0.30,
        "GLD":  0.20,
    },
    "balanced": {
        "AAPL": 0.15,
        "MSFT": 0.15,
        "SPY":  0.25,
        "QQQ":  0.15,
        "TLT":  0.15,
        "GLD":  0.15,
    },
    "aggressive": {
        "AAPL": 0.20,
        "MSFT": 0.20,
        "SPY":  0.25,
        "QQQ":  0.20,
        "TLT":  0.05,
        "GLD":  0.10,
    },
}


# ---------- Load helpers ----------

def load_asset_stats(path: Optional[Path] = None) -> Dict[str, Dict[str, float]]:
    """
    Load asset-level statistics (annual_return, annual_volatility, sharpe)
    from asset_stats.json.
    """
    if path is None:
        path = JSON_DIR / "asset_stats.json"

    if not path.exists():
        raise FileNotFoundError(f"asset_stats.json not found at {path}")

    with path.open("r", encoding="utf-8") as f:
        stats = json.load(f)

    return stats


def load_correlation_matrix(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load correlation matrix from correlation_matrix.json into a DataFrame.
    """
    if path is None:
        path = JSON_DIR / "correlation_matrix.json"

    if not path.exists():
        raise FileNotFoundError(f"correlation_matrix.json not found at {path}")

    with path.open("r", encoding="utf-8") as f:
        corr_dict = json.load(f)

    # corr_dict is nested: {ticker: {ticker: corr, ...}, ...}
    corr_df = pd.DataFrame(corr_dict)
    # Ensure symmetric order and all tickers included where possible
    # (if something missing, pandas will put NaN, we handle later)
    corr_df = corr_df.reindex(index=TICKERS, columns=TICKERS)
    return corr_df


def load_forecasts(path: Optional[Path] = None) -> Dict[str, Dict[str, float]]:
    """
    Load ML-enhanced forecasts from forecasts.json.

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

    if not path.exists():
        raise FileNotFoundError(f"forecasts.json not found at {path}")

    with path.open("r", encoding="utf-8") as f:
        forecasts = json.load(f)

    return forecasts


# ---------- Portfolio math ----------

def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Ensure weights sum to 1. If total is 0, return equal weights.
    """
    total = sum(max(w, 0.0) for w in weights.values())
    if total <= 0:
        # fallback: equal weights for non-negative entries
        n = len(weights)
        if n == 0:
            return {}
        equal_w = 1.0 / n
        return {k: equal_w for k in weights.keys()}

    return {k: max(w, 0.0) / total for k, w in weights.items()}


def compute_portfolio_stats(
    weights: Dict[str, float],
    asset_expected_returns: Dict[str, float],
    asset_vols: Dict[str, float],
    corr_matrix: pd.DataFrame,
) -> Dict[str, float]:
    """
    Compute portfolio expected return and risk.

    Args:
        weights: dict[ticker] -> weight (should sum ~1)
        asset_expected_returns: dict[ticker] -> expected 30d return
        asset_vols: dict[ticker] -> 30d volatility (std dev)
        corr_matrix: DataFrame of correlations between asset returns.

    Returns:
        {
          "expected_return": <float>,   # 30d expected portfolio return
          "expected_risk": <float>      # portfolio std dev (based on 30d vols)
        }
    """
    # Normalize weights to sum 1
    weights = _normalize_weights(weights)

    tickers = list(weights.keys())

    # Build vectors in this ticker order
    w = np.array([weights[t] for t in tickers], dtype=float)

    mu = np.array(
        [
            asset_expected_returns.get(t, 0.0)
            if asset_expected_returns.get(t) is not None
            else 0.0
            for t in tickers
        ],
        dtype=float,
    )

    sigma = np.array(
        [
            asset_vols.get(t, 0.0)
            if asset_vols.get(t) is not None
            else 0.0
            for t in tickers
        ],
        dtype=float,
    )

    # Correlation submatrix for the involved tickers
    try:
        corr_sub = corr_matrix.loc[tickers, tickers].to_numpy(dtype=float)
    except Exception:
        # If something goes wrong, fall back to identity
        corr_sub = np.eye(len(tickers), dtype=float)

    # Replace NaNs in correlation with 0 (uncorrelated approx), diagonal with 1
    if np.isnan(corr_sub).any():
        # Set diagonal to 1, others NaN->0
        for i in range(corr_sub.shape[0]):
            for j in range(corr_sub.shape[1]):
                if i == j:
                    corr_sub[i, j] = 1.0
                elif math.isnan(corr_sub[i, j]):
                    corr_sub[i, j] = 0.0

    # Covariance matrix: Σ = D * Corr * D, where D = diag(sigma)
    D = np.diag(sigma)
    cov = D @ corr_sub @ D

    # Expected portfolio return: w^T mu
    expected_return = float(np.dot(w, mu))

    # Portfolio variance: w^T Σ w
    var = float(w @ cov @ w)
    expected_risk = float(math.sqrt(max(var, 0.0)))

    return {
        "expected_return": expected_return,
        "expected_risk": expected_risk,
    }


# ---------- Forecast-based weight adjustment ----------

def adjust_weights_based_on_forecasts(
    profile_name: str,
    base_weights: Dict[str, float],
    asset_expected_returns: Dict[str, float],
) -> Dict[str, float]:
    """
    Start from BASE_WEIGHTS[profile], then:
      - Compute average expected 30d return across assets in this profile.
      - For assets with expected_return > avg: increase weight slightly.
      - For assets with expected_return < avg: decrease weight slightly.
      - Renormalize weights to sum 1.

    The magnitude of adjustment depends on profile (aggressive adjusts more).
    """
    weights = base_weights.copy()

    # Extract returns for tickers in this profile
    rets = []
    for t in weights.keys():
        r = asset_expected_returns.get(t)
        if r is not None:
            rets.append(r)

    if len(rets) == 0:
        # No forecast info, return base weights
        return _normalize_weights(weights)

    avg_ret = float(np.mean(rets))

    # Adjustment strength
    if profile_name == "conservative":
        delta = 0.02
    elif profile_name == "balanced":
        delta = 0.03
    else:  # aggressive
        delta = 0.04

    for t in list(weights.keys()):
        r = asset_expected_returns.get(t)
        if r is None:
            # No forecast info: leave as is
            continue

        if r > avg_ret:
            weights[t] += delta
        elif r < avg_ret:
            weights[t] -= delta
        # if equal-ish, leave unchanged

    # Normalize to ensure sum=1 and no negatives
    weights = _normalize_weights(weights)
    return weights


# ---------- Build all profile recommendations ----------

def build_recommended_portfolios() -> Dict[str, Dict[str, object]]:
    """
    Build recommended portfolios for each risk profile.

    Returns:
        {
          "conservative": {
            "weights": {ticker: weight, ...},
            "expected_return": <float>,
            "expected_risk": <float>
          },
          "balanced": { ... },
          "aggressive": { ... }
        }
    """
    asset_stats = load_asset_stats()
    corr_matrix = load_correlation_matrix()
    forecasts = load_forecasts()

    # Build asset_expected_returns and asset_vols from forecasts.json
    asset_expected_returns: Dict[str, float] = {}
    asset_vols: Dict[str, float] = {}

    for t in TICKERS:
        f = forecasts.get(t, {})
        exp_ret = f.get("expected_return_30d")
        exp_vol = f.get("expected_vol_30d")

        asset_expected_returns[t] = (
            float(exp_ret) if exp_ret is not None else 0.0
        )
        asset_vols[t] = (
            float(exp_vol) if exp_vol is not None else 0.0
        )

    recommendations: Dict[str, Dict[str, object]] = {}

    for profile_name, base_w in BASE_WEIGHTS.items():
        # Step 1: adjust weights based on forecasts
        adj_weights = adjust_weights_based_on_forecasts(
            profile_name,
            base_w,
            asset_expected_returns,
        )

        # Step 2: compute portfolio stats
        stats = compute_portfolio_stats(
            adj_weights,
            asset_expected_returns,
            asset_vols,
            corr_matrix,
        )

        recommendations[profile_name] = {
            "weights": adj_weights,
            "expected_return": stats["expected_return"],
            "expected_risk": stats["expected_risk"],
        }

    return recommendations


def save_portfolio_recommendations_json(
    recommendations: Dict[str, Dict[str, object]],
    path: Optional[Path] = None,
) -> Path:
    """
    Save portfolio recommendations to JSON.

    Schema:
        {
          "conservative": {
            "weights": {...},
            "expected_return": <float>,
            "expected_risk": <float>
          },
          "balanced": { ... },
          "aggressive": { ... }
        }
    """
    if path is None:
        path = JSON_DIR / "portfolio_recommendations.json"

    JSON_DIR.mkdir(parents=True, exist_ok=True)

    # Convert to pure Python floats for JSON (avoid numpy types)
    serializable = {}
    for profile, data in recommendations.items():
        weights = {
            t: float(w) for t, w in data["weights"].items()
        }
        serializable[profile] = {
            "weights": weights,
            "expected_return": float(data["expected_return"]),
            "expected_risk": float(data["expected_risk"]),
        }

    with path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    return path


# ---------- CLI entrypoint ----------

if __name__ == "__main__":
    recs = build_recommended_portfolios()
    out_path = save_portfolio_recommendations_json(recs)
    print(f"portfolio.py: portfolio recommendations saved to {out_path}")