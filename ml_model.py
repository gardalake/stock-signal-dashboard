# Version: v1.5.0

import pandas as pd

def train_predictive_model(data):
    if data is None or data.empty or "Close" not in data.columns:
        return "HOLD", 0.0

    try:
        data["SMA5"] = data["Close"].rolling(window=5).mean()
        data["SMA20"] = data["Close"].rolling(window=20).mean()

        if len(data.dropna()) < 1:
            return "HOLD", data["Close"].iloc[-1]

        last_close = data["Close"].iloc[-1]
        last_sma5 = data["SMA5"].iloc[-1]
        last_sma20 = data["SMA20"].iloc[-1]

        predicted_price = last_close * (1 + ((last_sma5 - last_sma20) / last_close))

        if last_sma5 > last_sma20:
            return "BUY", predicted_price
        elif last_sma5 < last_sma20:
            return "SELL", predicted_price
        else:
            return "HOLD", predicted_price
    except Exception:
        return "HOLD", data["Close"].iloc[-1]
