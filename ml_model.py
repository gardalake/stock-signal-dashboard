# Version: v1.6.1

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def compute_indicators(df):
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    low_14 = df["Low"].rolling(window=14).min()
    high_14 = df["High"].rolling(window=14).max()
    df["StochRSI"] = (df["Close"] - low_14) / (high_14 - low_14)
    df["Momentum"] = df["Close"] - df["Close"].shift(4)
    df.dropna(inplace=True)
    return df

def train_predictive_model(df):
    df = compute_indicators(df)

    features = ["MA20", "MA50", "RSI", "StochRSI", "Momentum"]
    predictions = {}
    live_price = df["Close"].iloc[-1]

    for horizon in [1, 3, 5, 7]:
        df[f"target_{horizon}d"] = df["Close"].shift(-horizon)

        data = df.dropna().copy()
        X = data[features]
        y = data[f"target_{horizon}d"]

        if len(X) < 30:
            predictions[f"{horizon}d"] = np.nan
            continue

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_pred_scaled = scaler.transform([X.iloc[-1]])

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_pred_scaled)[0]
        predictions[f"{horizon}d"] = float(y_pred)

    # Determina il segnale principale su 3d
    target_3d = predictions.get("3d", live_price)
    change_pct = (target_3d - live_price) / live_price * 100
    if change_pct > 0.5:
        ai_signal = "BUY"
    elif change_pct < -0.5:
        ai_signal = "SELL"
    else:
        ai_signal = "HOLD"

    return {
        "ai_signal": ai_signal,
        "live_price": live_price,
        "predictions": predictions
    }
