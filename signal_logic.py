# Version: v1.5.0

import pandas as pd
from ml_model import train_predictive_model
from email_utils import send_signal_email
import yaml

try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
except Exception:
    config = {"send_email": False}

def detect_breakout(data):
    if len(data) < 20:
        return None

    high = data["High"].rolling(window=20).max()
    low = data["Low"].rolling(window=20).min()
    volume_avg = data["Volume"].rolling(window=20).mean()

    last_close = data["Close"].iloc[-1]
    last_high = high.iloc[-2]
    last_low = low.iloc[-2]
    last_volume = data["Volume"].iloc[-1]
    last_volume_avg = volume_avg.iloc[-2]

    if last_close > last_high and last_volume > last_volume_avg:
        return "BULLISH_BREAKOUT", last_high
    elif last_close < last_low and last_volume > last_volume_avg:
        return "BEARISH_BREAKOUT", last_low
    return None, None

def generate_signals(data, symbol):
    if data is None or data.empty:
        return {
            "ai_signal": "NO_DATA",
            "predicted_price": 0.0
        }

    ai_signal, predicted_price = train_predictive_model(data)
    breakout_signal, breakout_level = detect_breakout(data)

    if breakout_signal == "BULLISH_BREAKOUT":
        ai_signal = "BUY"
    elif breakout_signal == "BEARISH_BREAKOUT":
        ai_signal = "SELL"

    if config.get("send_email", False) and ai_signal in ["BUY", "SELL"]:
        send_signal_email(symbol, ai_signal, predicted_price)

    return {
        "ai_signal": ai_signal,
        "predicted_price": predicted_price,
        "breakout_level": breakout_level if breakout_level else None
    }
