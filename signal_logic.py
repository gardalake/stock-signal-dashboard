# Version: v1.6.0

import pandas as pd
from ml_model import train_predictive_model
from email_utils import send_signal_email
import yaml

# Load config
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
except:
    config = {
        "email_notifications": False
    }

def generate_signals(data, ticker):
    signal_data = train_predictive_model(data)
    ai_signal = signal_data["ai_signal"]
    live_price = signal_data["live_price"]
    predictions = signal_data["predictions"]

    result = {
        "ai_signal": ai_signal,
        "live_price": live_price,
        "predictions": predictions,
    }

    # Optional: breakout detection (simple high of last 14 days)
    try:
        breakout_level = data["High"].rolling(window=14).max().iloc[-1]
        result["breakout_level"] = breakout_level
    except:
        result["breakout_level"] = None

    # Optional email notifications
    if config.get("email_notifications", False):
        send_signal_email(ticker, ai_signal, live_price, predictions)

    return result
