# 22CSE633 — Applied Time Series Analysis

**Satya Sankeerth Thalikota | 23WU0102182 | AIML Panthers**
Exchange Student Evaluation — Unmapped Course

## Contents

- `assignment1/sarima_airpassengers.py` — Assignment 1: stationarity, ADF test,
  ACF/PACF, SARIMA order selection via AIC grid search, walk-forward
  validation, and residual diagnostics on the AirPassengers dataset.
- `assignment2/lstm_stock_forecast.py` — Assignment 2: LSTM one-step-ahead
  forecaster on daily closing prices, evaluated against a naive persistence
  baseline.

## Running

Each assignment folder has its own `requirements.txt`.

```bash
cd assignment1
pip install -r requirements.txt
python sarima_airpassengers.py

cd ../assignment2
pip install -r requirements.txt
python lstm_stock_forecast.py --ticker AAPL --start 2019-01-01 --end 2023-01-01
# or, without internet access:
python lstm_stock_forecast.py --offline
```

Plots and metrics are written to an `outputs/` folder created inside each
assignment directory.

## Reports

The written reports (PDF/Word, with plots, equations, and analysis) are
submitted separately and reference the code in this repository.
