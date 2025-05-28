# Version: v1.4.2

import pandas as pd
from ml_model import train_predictive_model
from email_utils import send_signal_email
import yaml

# Load config
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
except Exception:
    config = {"send_email": False}

def generate_signals(data, symbol):
    if data is None or data.empty:
        return {
            "ai_signal": "NO_DATA",
            "predicted_price": 0.0
        }

    ai_signal, predicted_price = train_predictive_model(data)

    if config.get("send_email", False) and ai_signal in ["BUY", "SELL"]:
        send_signal_email(symbol, ai_signal, predicted_price)

    return {
        "ai_signal": ai_signal,
        "predicted_price": predicted_price
    }
