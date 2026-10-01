"""
22CSE633 - Applied Time Series Analysis
Assignment 2: AI-based Forecasting & Applied Time Series Domains
Satya Sankeerth Thalikota | 23WU0102182 | AIML Panthers

Domain: financial time series (daily closing price), one-step-ahead
forecasting with an LSTM, compared against a naive persistence baseline.

By default this pulls a real ticker with yfinance. If yfinance / internet
is not available, it falls back to a simulated GARCH(1,1)-style closing
price series so the rest of the pipeline still runs end-to-end offline.

Requirements:
    pip install numpy pandas matplotlib scikit-learn tensorflow yfinance

Run:
    python lstm_stock_forecast.py --ticker AAPL --start 2019-01-01 --end 2023-01-01
    python lstm_stock_forecast.py --offline          # use simulated data instead
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

WINDOW = 20  # trading days used to predict the next day's close


# ---------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------
def load_real_data(ticker, start, end):
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end)
    return df["Close"].dropna()


def simulate_data(n=1000, seed=42):
    """GARCH(1,1)-style synthetic closing price, used only as an offline
    fallback when a live data pull is not available."""
    rng = np.random.default_rng(seed)
    omega, alpha1, beta1 = 1e-6, 0.08, 0.90
    sigma2 = np.zeros(n); eps = np.zeros(n); ret = np.zeros(n)
    sigma2[0] = omega / (1 - alpha1 - beta1)
    for t in range(1, n):
        sigma2[t] = omega + alpha1 * eps[t - 1] ** 2 + beta1 * sigma2[t - 1]
        eps[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
        ret[t] = 0.0003 + eps[t]
    price = 100 * np.exp(np.cumsum(ret))
    dates = pd.bdate_range("2019-01-01", periods=n)
    return pd.Series(price, index=dates, name="Close")


def make_windows(arr, window):
    X, y = [], []
    for i in range(window, len(arr)):
        X.append(arr[i - window:i])
        y.append(arr[i])
    return np.array(X), np.array(y)


def metrics(actual, pred):
    mae = np.mean(np.abs(actual - pred))
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    return mae, rmse, mape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2023-01-01")
    ap.add_argument("--offline", action="store_true", help="use simulated data")
    args = ap.parse_args()

    if args.offline:
        close = simulate_data()
        print("Using simulated GARCH(1,1)-style closing price series (offline mode).")
    else:
        try:
            close = load_real_data(args.ticker, args.start, args.end)
            print(f"Downloaded {len(close)} days of {args.ticker} closing prices.")
        except Exception as e:
            print(f"Live download failed ({e}); falling back to simulated data.")
            close = simulate_data()

    close.to_csv(f"{OUT}/close_prices.csv")
    ret = np.log(close).diff().dropna()

    fig, axes = plt.subplots(2, 1, figsize=(9, 5))
    axes[0].plot(close); axes[0].set_title("Daily Closing Price")
    axes[1].plot(ret, color="tab:red", lw=0.6); axes[1].set_title("Daily Log Returns")
    plt.tight_layout(); plt.savefig(f"{OUT}/1_raw.png", dpi=150); plt.close()

    # -------------------------------------------------------------
    # 2. Windowing + chronological train/val/test split (no leakage)
    # -------------------------------------------------------------
    n = len(close)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    values = close.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler.fit(values[:n_train])                      # fit ONLY on training data
    scaled = scaler.transform(values).flatten()

    X_all, y_all = make_windows(scaled, WINDOW)
    split1 = n_train - WINDOW
    split2 = n_train + n_val - WINDOW

    X_train, y_train = X_all[:split1], y_all[:split1]
    X_val, y_val = X_all[split1:split2], y_all[split1:split2]
    X_test, y_test = X_all[split2:], y_all[split2:]

    X_train = X_train.reshape((-1, WINDOW, 1))
    X_val = X_val.reshape((-1, WINDOW, 1))
    X_test = X_test.reshape((-1, WINDOW, 1))
    print(f"Train/Val/Test windows: {len(X_train)}/{len(X_val)}/{len(X_test)}")

    # -------------------------------------------------------------
    # 3. LSTM model
    # -------------------------------------------------------------
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=(WINDOW, 1)),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=60, batch_size=32, verbose=1,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)],
    )

    plt.figure(figsize=(9, 3.5))
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.legend(); plt.title("Training loss curve")
    plt.tight_layout(); plt.savefig(f"{OUT}/2_loss.png", dpi=150); plt.close()

    # -------------------------------------------------------------
    # 4. Evaluate vs. naive persistence baseline
    # -------------------------------------------------------------
    pred_test_scaled = model.predict(X_test).flatten()
    pred_test = scaler.inverse_transform(pred_test_scaled.reshape(-1, 1)).flatten()
    actual_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    naive_test = scaler.inverse_transform(X_test[:, -1, 0].reshape(-1, 1)).flatten()

    mae_m, rmse_m, mape_m = metrics(actual_test, pred_test)
    mae_n, rmse_n, mape_n = metrics(actual_test, naive_test)
    print(f"\nLSTM  : MAE={mae_m:.3f} RMSE={rmse_m:.3f} MAPE={mape_m:.2f}%")
    print(f"Naive : MAE={mae_n:.3f} RMSE={rmse_n:.3f} MAPE={mape_n:.2f}%")

    test_dates = close.index[WINDOW + split2: WINDOW + split2 + len(y_test)]
    plt.figure(figsize=(9, 4))
    plt.plot(test_dates, actual_test, label="Actual")
    plt.plot(test_dates, pred_test, "--", label="LSTM forecast")
    plt.plot(test_dates, naive_test, ":", label="Naive (persistence)")
    plt.legend(); plt.title(f"Test-set forecast | LSTM RMSE={rmse_m:.2f} vs Naive RMSE={rmse_n:.2f}")
    plt.tight_layout(); plt.savefig(f"{OUT}/3_forecast.png", dpi=150); plt.close()

    print("\nAll plots saved to ./outputs/")


if __name__ == "__main__":
    main()
