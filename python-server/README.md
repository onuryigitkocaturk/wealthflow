# WealthFlow – Python ML Engine

This folder contains the Python + ML part of the **WealthFlow** FinTech project.

Core idea: an AI-powered robo-advisor that uses public market data (AAPL, MSFT, SPY,
QQQ, TLT, GLD), computes risk metrics, builds baseline + ARIMA + LSTM models, and
produces portfolio recommendations for different risk profiles.

## Project structure

- `server/data/raw/` – raw CSV files from Investing.com (`AAPL.csv`, `MSFT.csv`, ...)
- `server/data/processed/` – cleaned CSV files (created automatically)
- `server/src/data_loading.py` – data cleaning & loading helpers
- `server/src/features.py` – feature engineering, risk metrics, EDA figures
- `server/src/models_baseline.py` – naive & moving-average baselines
- `server/src/models_ml.py` – ARIMA + LSTM models, ML-based forecasts
- `server/src/portfolio.py` – portfolio construction & robo-advisor logic
- `server/outputs/json/` – JSON outputs (risk metrics, forecasts, portfolios)
- `server/outputs/figures/` – plots for the report
- `server/notebooks/` – Jupyter notebooks (EDA, models)

## 1. Setup

### Clone repository

```bash
git clone https://github.com/onuryigitkocaturk/wealthflow.git
cd wealthflow/python-server
```

### Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # Mac / Linux
# .venv\Scripts\activate         # Windows (PowerShell / CMD)
```

### Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Copy the 6 CSV files exported from Investing.com into:
server/data/raw/
AAPL.csv
MSFT.csv
SPY.csv
QQQ.csv
TLT.csv
GLD.csv

## 2. Run pipeline

### From python-server/ (with the virtual env active):

```bash
# 1) Compute risk metrics, descriptive stats, missing/outlier reports, and plots
python server/src/features.py

# 2) Build baseline 30-day forecasts
python server/src/models_baseline.py

# 3) Train ARIMA + LSTM models and build ML-based forecasts
python server/src/models_ml.py

# 4) Build portfolio recommendations (conservative / balanced / aggressive)
python server/src/portfolio.py
```

### After these steps, you should have:
• server/outputs/json/asset_stats.json
• server/outputs/json/correlation_matrix.json
• server/outputs/json/descriptive_stats_returns.json
• server/outputs/json/missing_report.json
• server/outputs/json/outlier_report.json
• server/outputs/json/forecasts_baseline.json
• server/outputs/json/forecasts.json
• server/outputs/json/portfolio_recommendations.json

and multiple plots under server/outputs/figures/.

## 3. Notebooks

### To explore the results interactively:

```bash
jupyter notebook server/notebooks/01_eda.ipynb
jupyter notebook server/notebooks/02_models.ipynb
```

### The notebooks demonstrate:

    •	EDA of price and return series
    •	Comparison of Naive / MovingAverage / ARIMA / LSTM on SPY
    •	How forecasts feed into the portfolio engine
    •	Risk–return profiles for different investor types
