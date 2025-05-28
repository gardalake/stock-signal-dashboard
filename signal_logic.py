# Version: v1.4.0

import pandas as pd
from ml_model import train_predictive_model
from email_utils import send_signal_email
import yaml

# Load config
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        stock_spread_pct = config.get("etoro_spread_pct", 0.6) / 100.0
except:
    stock_spread_pct = 0.006  # default 0.6% if config not found

crypto_spread_pct = 0.025  # 2.5% for crypto by default

def is_crypto(ticker):
    return ticker.endswith("-USD")

def detect_breakout(data, window=10, volume_ma=5):
    data = data.copy()
    data["HighMax"] = data["High"].rolling(window=window).max()
    data["LowMin"] = data["Low"].rolling(window=window).min()
    data["VolMA"] = data["Volume"].rolling(volume_ma).mean()
    breakout_signals = []

    for i in range(window, len(data)):
        row = data.iloc[i]
        prev_row = data.iloc[i - 1]

        # Bullish Breakout
        if row["Close"] > prev_row["HighMax"] and row["Volume"] > row["VolMA"]:
            breakout_signals.append({"date": data.index[i], "signal": "BREAKOUT ↑", "price": row["Close"]})

        # Bearish Breakout
        elif row["Close"] < prev_row["LowMin"] and row["Volume"] > row["VolMA"]:
            breakout_signals.append({"date": data.index[i], "signal": "BREAKOUT ↓", "price": row["Close"]})

    return breakout_signals

def generate_signals(data, ticker="UNKNOWN"):
    signals = []
    spread_pct = crypto_spread_pct if is_crypto(ticker) else stock_spread_pct

    data["MA50"] = data["Close"].rolling(window=50).mean()
    data["RSI"] = 100 - (100 / (1 + data["Close"].pct_change().rolling(14).mean()))

    for i in range(1, len(data)):
        price = data['Close'].iloc[i]
        ma50 = data["MA50"].iloc[i]
        rsi = data["RSI"].iloc[i]

        if rsi < 30 and price < ma50:
            next_price = data["Close"].iloc[i+1] if i+1 < len(data) else price
            gain = (next_price - price) / price
            if gain > spread_pct:
                signals.append({"date": data.index[i], "signal": "BUY", "price": price})

        elif rsi > 70 and price > ma50:
            prev_price = data["Close"].iloc[i-1]
            drop = (price - prev_price) / prev_price
            if drop > spread_pct:
                signals.append({"date": data.index[i], "signal": "SELL", "price": price})

    # AI Signal
    ai_signal, predicted_price = train_predictive_model(data)
    if ai_signal != "HOLD":
        today = data.index[-1]
        current_price = data["Close"].iloc[-1]
        change = abs(predicted_price - current_price) / current_price

        if change > spread_pct:
            signals.append({"date": today, "signal": f"AI {ai_signal}", "price": current_price})
            subject = f"📢 Signal Alert: {ai_signal} on {ticker}"
            body = f"Signal: {ai_signal}\nCurrent Price: {current_price:.2f}\nPredicted Price: {predicted_price:.2f}\nDate: {today.strftime('%Y-%m-%d')}"
            send_signal_email(subject, body)

    # Breakout Signals
    breakout_signals = detect_breakout(data)
    signals.extend(breakout_signals)

    return signals
