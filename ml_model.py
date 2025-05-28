# Version: v1.4.0

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def train_predictive_model(data):
    data = data.copy()
    data["MA50"] = data["Close"].rolling(window=50).mean()
    data["MA200"] = data["Close"].rolling(window=200).mean()
    data["Momentum"] = data["Close"] - data["Close"].shift(1)
    data.dropna(inplace=True)

    features = data[["MA50", "MA200", "Momentum"]]
    target = data["Close"].shift(-1)
    features.dropna(inplace=True)
    target = target.loc[features.index]

    if len(features) < 20:
        return "HOLD", data["Close"].iloc[-1]

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    prediction = model.predict([features.iloc[-1]])[0]
    current_price = data["Close"].iloc[-1]
    change = (prediction - current_price) / current_price

    if change > 0.015:
        return "BUY", prediction
    elif change < -0.015:
        return "SELL", prediction
    else:
        return "HOLD", prediction
