"""
22CSE633 - Applied Time Series Analysis
Assignment 1: Foundations of Time Series & Traditional Forecasting Models
Satya Sankeerth Thalikota | 23WU0102182 | AIML Panthers

Dataset: classic AirPassengers dataset (Box & Jenkins, 1976) - monthly
totals of international airline passengers (thousands), Jan 1949-Dec 1960.

Requirements:
    pip install pandas numpy matplotlib statsmodels scikit-learn

Run:
    python sarima_airpassengers.py
Outputs (saved to ./outputs/):
    raw series, log series, differenced series, ACF/PACF, AIC comparison,
    walk-forward forecast plot, residual ACF plot, and printed metrics.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
DATA = [
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
    115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
    145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
    171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
    196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
    204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
    242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
    284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
    315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
    340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
    360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
    417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432,
]
idx = pd.date_range("1949-01-01", periods=len(DATA), freq="MS")
series = pd.Series(DATA, index=idx, name="Passengers")
series.to_csv(os.path.join(OUT, "airpassengers.csv"))
print(f"Loaded {len(series)} monthly observations.")

# ---------------------------------------------------------------------
# 2. Raw plot + log transform (stabilise multiplicative seasonality)
# ---------------------------------------------------------------------
plt.figure(figsize=(9, 4))
plt.plot(series)
plt.title("Monthly International Airline Passengers (1949-1960)")
plt.xlabel("Year"); plt.ylabel("Passengers (thousands)")
plt.tight_layout(); plt.savefig(f"{OUT}/1_raw.png", dpi=150); plt.close()

log_y = np.log(series)
plt.figure(figsize=(9, 4))
plt.plot(log_y, color="tab:red")
plt.title("Log-Transformed Series")
plt.tight_layout(); plt.savefig(f"{OUT}/2_log.png", dpi=150); plt.close()

# ---------------------------------------------------------------------
# 3. Stationarity: ADF test, then d=1 + D=1(s=12) differencing
# ---------------------------------------------------------------------
adf_level = adfuller(log_y)
print(f"\nADF on log(level):      stat={adf_level[0]:.3f}  p-value={adf_level[1]:.4f}")

z = log_y.diff(1).diff(12).dropna()
adf_diff = adfuller(z)
print(f"ADF on differenced series: stat={adf_diff[0]:.3f}  p-value={adf_diff[1]:.4f}")

plt.figure(figsize=(9, 3.5))
plt.plot(z, color="tab:green"); plt.axhline(0, color="k", lw=0.7)
plt.title("Series after d=1 (non-seasonal) and D=1 (seasonal, s=12) differencing")
plt.tight_layout(); plt.savefig(f"{OUT}/3_differenced.png", dpi=150); plt.close()

# ---------------------------------------------------------------------
# 4. ACF / PACF of the differenced series
# ---------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(9, 6))
plot_acf(z, lags=24, ax=axes[0], title="ACF of differenced series")
plot_pacf(z, lags=24, ax=axes[1], method="ywm", title="PACF of differenced series")
plt.tight_layout(); plt.savefig(f"{OUT}/4_acf_pacf.png", dpi=150); plt.close()

# ---------------------------------------------------------------------
# 5. Order selection: grid search (p,d,q)(P,D,Q,12) scored by AIC
#    d=1, D=1, s=12 fixed (justified by the ADF tests above)
# ---------------------------------------------------------------------
train, val = log_y.iloc[:-24], log_y.iloc[-24:]

candidates = []
for p in range(4):
    for q in range(2):
        for P in range(2):
            for Q in range(2):
                if p == 0 and q == 0:
                    continue
                try:
                    fit = SARIMAX(
                        train, order=(p, 1, q), seasonal_order=(P, 1, Q, 12),
                        enforce_stationarity=False, enforce_invertibility=False,
                    ).fit(disp=False)
                    candidates.append(((p, 1, q, P, 1, Q, 12), fit.aic))
                except Exception:
                    continue

candidates.sort(key=lambda c: c[1])
print("\nTop 5 candidate orders by AIC:")
for order, aic in candidates[:5]:
    p, d, q, P, D, Q, s = order
    print(f"  SARIMA({p},{d},{q})({P},{D},{Q},{s})  AIC={aic:.2f}")

best_order = candidates[0][0]
p, d, q, P, D, Q, s = best_order
print(f"\nSelected: SARIMA({p},{d},{q})({P},{D},{Q},{s})")

labels = [f"({o[0]},1,{o[2]})({o[3]},1,{o[5]},12)" for o, _ in candidates[:8]]
aics = [a for _, a in candidates[:8]]
plt.figure(figsize=(9, 4))
colors = ["tab:green" if i == 0 else "tab:blue" for i in range(len(aics))]
plt.bar(labels, aics, color=colors)
plt.xticks(rotation=35, ha="right"); plt.ylabel("AIC")
plt.title("AIC across candidate SARIMA orders (lower is better; best in green)")
plt.tight_layout(); plt.savefig(f"{OUT}/5_aic_comparison.png", dpi=150); plt.close()

# ---------------------------------------------------------------------
# 6. Fit best model, walk-forward one-step-ahead validation
# ---------------------------------------------------------------------
fit = SARIMAX(
    train, order=(p, d, q), seasonal_order=(P, D, Q, s),
    enforce_stationarity=False, enforce_invertibility=False,
).fit(disp=False)
print(fit.summary())

history = train.copy()
preds = []
for t in val.index:
    step = SARIMAX(history, order=(p, d, q), seasonal_order=(P, D, Q, s)).filter(fit.params)
    preds.append(step.forecast(1).iloc[0])
    history = pd.concat([history, val.loc[[t]]])

preds = np.exp(preds)
actual = np.exp(val)
rmse = np.sqrt(mean_squared_error(actual, preds))
mae = mean_absolute_error(actual, preds)
mape = np.mean(np.abs((actual - preds) / actual)) * 100
print(f"\nWalk-forward validation (last 24 months): MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%")

plt.figure(figsize=(9, 4))
plt.plot(series, color="gray", label="Full series")
plt.plot(val.index, actual, "o-", label="Actual (validation)")
plt.plot(val.index, preds, "s--", color="tab:red", label="Forecast (walk-forward)")
plt.legend(); plt.title(f"Walk-forward forecast  |  RMSE={rmse:.1f}, MAPE={mape:.2f}%")
plt.tight_layout(); plt.savefig(f"{OUT}/6_forecast.png", dpi=150); plt.close()

# ---------------------------------------------------------------------
# 7. Residual diagnostics
# ---------------------------------------------------------------------
resid = fit.resid[fit.loglikelihood_burn:]
lb = acorr_ljungbox(resid, lags=[10], return_df=True)
print(f"\nLjung-Box Q(10)={lb['lb_stat'].iloc[0]:.2f}  p-value={lb['lb_pvalue'].iloc[0]:.4f}")

fig, ax = plt.subplots(figsize=(9, 3.5))
plot_acf(resid, lags=20, ax=ax, title="Residual ACF (in-sample)")
plt.tight_layout(); plt.savefig(f"{OUT}/7_residual_acf.png", dpi=150); plt.close()

print("\nAll plots saved to ./outputs/")
