# Version: v1.6.2

import pandas as pd
from ml_model import train_predictive_model
from email_utils import send_signal_email
import yaml

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

SPREADS = config.get("spreads", {})

def apply_spread(price, ticker):
    if any(ticker.upper().startswith(c) for c in ["BTC", "ETH", "SOL", "RNDR"]):
        spread = SPREADS.get("crypto", 0.025)
    else:
        spread = SPREADS.get("stock", 0.006)
    return price * (1 + spread)

def generate_signals(df, ticker):
    model_result = train_predictive_model(df)
    signal = model_result["ai_signal"]
    live_price = model_result["live_price"]
    predictions = model_result["predictions"]

    # Applica spread e arrotonda
    adjusted_price = apply_spread(live_price, ticker)
    formatted_predictions = {}
    for k, v in predictions.items():
        if v is not None and not pd.isna(v):
            spreaded = apply_spread(v, ticker)
            change = (spreaded - adjusted_price) / adjusted_price * 100
            trend = "🔺" if change > 0 else "🔻"
            formatted_predictions[k] = f"{spreaded:.2f} {trend} ({change:+.2f}%)"
        else:
            formatted_predictions[k] = "N/A"

    signal_output = {
        "ai_signal": signal,
        "live_price": f"{adjusted_price:.2f}",
        "predictions": formatted_predictions
    }

    if config.get("email", {}).get("enabled", False):
        send_signal_email(ticker, signal_output)

    return signal_output
