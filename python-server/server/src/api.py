"""
FastAPI API layer for WealthFlow robo-advisor.

Single business endpoint:
    POST /allocation

This endpoint:
- Reads precomputed JSON outputs (risk metrics, ML forecasts, portfolio recommendations)
- Returns a full portfolio allocation for a given risk profile and investment amount.

Assumptions:
- You have already run (from python-server root):
    python server/src/features.py
    python server/src/models_baseline.py
    python server/src/models_ml.py
    python server/src/portfolio.py

Directory structure (from python-server root):
- server/outputs/json/asset_stats.json
- server/outputs/json/forecasts.json
- server/outputs/json/portfolio_recommendations.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ---------- Constants (mirror of data_loading.TICKERS) ----------

TICKERS = ["AAPL", "MSFT", "SPY", "QQQ", "TLT", "GLD"]


# ---------- Paths & global state ----------

PYTHON_ROOT = Path(__file__).resolve().parents[1]  # .../python-server/server
OUTPUTS_DIR = PYTHON_ROOT / "outputs"
JSON_DIR = OUTPUTS_DIR / "json"

ASSET_STATS: Dict[str, Any] = {}
FORECASTS_ML: Dict[str, Any] = {}
PORTFOLIO_RECOMMENDATIONS: Dict[str, Any] = {}


def _load_json_file(filename: str) -> Dict[str, Any]:
    """Load a JSON file from JSON_DIR, or raise a helpful error."""
    path = JSON_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _init_data() -> None:
    """Load all required JSON files into global variables at startup."""
    global ASSET_STATS, FORECASTS_ML, PORTFOLIO_RECOMMENDATIONS

    print(f"[WealthFlow API] Loading JSON files from: {JSON_DIR}")

    ASSET_STATS = _load_json_file("asset_stats.json")
    FORECASTS_ML = _load_json_file("forecasts.json")
    PORTFOLIO_RECOMMENDATIONS = _load_json_file("portfolio_recommendations.json")

    # Basic sanity logs
    for t in TICKERS:
        if t not in ASSET_STATS:
            print(f"[WARN] {t} missing from asset_stats.json")
        if t not in FORECASTS_ML:
            print(f"[WARN] {t} missing from forecasts.json")

    print("[WealthFlow API] JSON files loaded successfully.")


# ---------- Pydantic models ----------

class AllocationRequest(BaseModel):
    risk_profile: str = Field(
        ...,
        description="Risk profile: conservative | balanced | aggressive",
        examples=["conservative", "balanced", "aggressive"],
    )
    investment_amount: float = Field(
        ...,
        gt=0,
        description="Total investment amount (e.g., in USD).",
        examples=[10000.0],
    )


class AssetAllocation(BaseModel):
    weight: float
    amount: float
    expected_return_30d: float | None = None
    expected_vol_30d: float | None = None
    expected_profit_30d: float | None = None
    risk_metrics: Dict[str, float] | None = None


class PortfolioSummary(BaseModel):
    profile: str
    investment_amount: float
    expected_return_30d: float
    expected_risk_30d: float
    expected_profit_30d: float
    expected_value_30d: float


class AllocationResponse(BaseModel):
    profile: str
    investment_amount: float
    horizon_days: int
    portfolio_summary: PortfolioSummary
    allocations: Dict[str, AssetAllocation]


# ---------- FastAPI app ----------

app = FastAPI(
    title="WealthFlow Robo-Advisory API",
    description=(
        "AI-powered robo-advisory allocation service for the WealthFlow FinTech project.\n\n"
        "Single main endpoint:\n"
        "- POST /allocation: returns portfolio allocation for a given risk profile and investment amount.\n\n"
        "All heavy ML/ARIMA/LSTM computation is done offline by Python scripts."
    ),
    version="1.0.0",
)

# CORS: allow everything for now (you can restrict later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # iOS, Web, anything
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    """Load JSON data into memory when the API starts."""
    try:
        _init_data()
    except FileNotFoundError as e:
        # Fail fast if pipeline was not run
        print(f"[FATAL] {e}")
        # Let the exception propagate; uvicorn will show the traceback.


# ---------- /allocation endpoint ----------

@app.post("/allocation", response_model=AllocationResponse, summary="Get portfolio allocation")
def get_allocation(req: AllocationRequest) -> AllocationResponse:
    """
    Compute portfolio allocation for a given risk profile and investment amount.

    Logic:
    - Use portfolio_recommendations.json to get:
        - base weights per asset
        - portfolio-level expected_return and expected_risk (30-day horizon)
    - Use forecasts.json to enrich each asset with:
        - expected_return_30d, expected_vol_30d
    - Use asset_stats.json to attach risk metrics (annual_return, annual_volatility, sharpe)
    - Scale weights by investment_amount to compute dollar allocation per asset.
    """
    if not PORTFOLIO_RECOMMENDATIONS:
        raise HTTPException(
            status_code=500,
            detail="portfolio_recommendations.json not loaded. "
                   "Run the ML pipeline scripts before starting the API."
        )
    if not FORECASTS_ML or not ASSET_STATS:
        raise HTTPException(
            status_code=500,
            detail="Required JSON files (forecasts.json / asset_stats.json) not loaded."
        )

    profile_key = req.risk_profile.lower()
    if profile_key not in PORTFOLIO_RECOMMENDATIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid risk_profile '{req.risk_profile}'. "
                   f"Use one of: {list(PORTFOLIO_RECOMMENDATIONS.keys())}",
        )

    profile_data = PORTFOLIO_RECOMMENDATIONS[profile_key]
    weights_dict = profile_data.get("weights", {})

    if not weights_dict:
        raise HTTPException(
            status_code=500,
            detail=f"No weights found for profile '{profile_key}' in portfolio_recommendations.json",
        )

    horizon_days = 30  # by design, all forecasts are 30-day

    # Portfolio-level expected return and risk (from JSON, already aggregated)
    port_ret_30d = float(profile_data.get("expected_return", 0.0))
    port_risk_30d = float(profile_data.get("expected_risk", 0.0))

    investment = float(req.investment_amount)
    port_profit_30d = investment * port_ret_30d
    port_value_30d = investment * (1.0 + port_ret_30d)

    allocations: Dict[str, AssetAllocation] = {}

    for ticker, w in weights_dict.items():
        # ignore any asset that isn't in TICKERS (defensive)
        if ticker not in TICKERS:
            continue

        weight = float(w)
        amount = investment * weight

        forecast = FORECASTS_ML.get(ticker, {})
        risk = ASSET_STATS.get(ticker, {})

        exp_ret = forecast.get("expected_return_30d")
        exp_vol = forecast.get("expected_vol_30d")

        if exp_ret is not None:
            expected_profit_30d = amount * float(exp_ret)
        else:
            expected_profit_30d = None

        allocations[ticker] = AssetAllocation(
            weight=weight,
            amount=amount,
            expected_return_30d=float(exp_ret) if exp_ret is not None else None,
            expected_vol_30d=float(exp_vol) if exp_vol is not None else None,
            expected_profit_30d=expected_profit_30d,
            risk_metrics=risk if risk else None,
        )

    portfolio_summary = PortfolioSummary(
        profile=profile_key,
        investment_amount=investment,
        expected_return_30d=port_ret_30d,
        expected_risk_30d=port_risk_30d,
        expected_profit_30d=port_profit_30d,
        expected_value_30d=port_value_30d,
    )

    return AllocationResponse(
        profile=profile_key,
        investment_amount=investment,
        horizon_days=horizon_days,
        portfolio_summary=portfolio_summary,
        allocations=allocations,
    )