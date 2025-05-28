# Version: v1.6.0

import pandas as pd
import numpy as np

def calculate_indicators(data):
    data = data.copy()
    data["MA20"] = data["Close"].rolling(window=20).mean()
    data["RSI"] = compute_rsi(data["Close"], 14)
    data["StochRSI"] = compute_stoch_rsi(data["RSI"])
    data.dropna(inplace=True)
    return data

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_stoch_rsi(rsi, period=14):
    min_val = rsi.rolling(window=period).min()
    max_val = rsi.rolling(window=period).max()
    return (rsi - min_val) / (max_val - min_val)

def predict_prices(data, horizons=[1, 3, 5, 7]):
    predictions = {}
    if data.empty or len(data) < 20:
        for h in horizons:
            predictions[f"{h}d"] = np.nan
        return predictions

    latest = data.iloc[-1]
    for h in horizons:
        trend = 0
        if latest["RSI"] < 30 and latest["StochRSI"] < 0.2:
            trend = 1
        elif latest["RSI"] > 70 and latest["StochRSI"] > 0.8:
            trend = -1

        drift = 0.002 * h
        prediction = latest["Close"] * (1 + drift * trend)
        predictions[f"{h}d"] = prediction
    return predictions

def train_predictive_model(data):
    data = calculate_indicators(data)
    predictions = predict_prices(data)
    latest_price = data["Close"].iloc[-1]

    # Decision rule
    target = predictions.get("1d", latest_price)
    change_pct = ((target - latest_price) / latest_price) * 100
    if change_pct > 0.5:
        signal = "BUY"
    elif change_pct < -0.5:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "ai_signal": signal,
        "live_price": latest_price,
        "predictions": predictions
    }
