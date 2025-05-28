# Version: v1.5.0

import pandas as pd
import requests
import time

def is_crypto(symbol):
    return symbol.endswith("-USD")

def get_crypto_data(symbol):
    symbol_map = {
        "BTC-USD": "bitcoin",
        "ETH-USD": "ethereum",
        "SOL-USD": "solana",
        "RNDR-USD": "render-token"
    }

    if symbol not in symbol_map:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    coin_id = symbol_map[symbol]
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": "90", "interval": "daily"}

    for attempt in range(3):
        try:
            r = requests.get(url, params=params)
            if r.status_code == 200:
                json_data = r.json()
                prices = json_data["prices"]
                df = pd.DataFrame(prices, columns=["timestamp", "price"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)
                df["Open"] = df["price"]
                df["High"] = df["price"]
                df["Low"] = df["price"]
                df["Close"] = df["price"]
                df["Volume"] = 1000000
                return df[["Open", "High", "Low", "Close", "Volume"]]
        except Exception as e:
            print(f"[CoinGecko] Attempt {attempt+1} failed: {e}")
        time.sleep(2)

    return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

def get_stock_data(symbol, api_key):
    if is_crypto(symbol):
        return get_crypto_data(symbol)

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "apikey": api_key,
        "outputsize": "compact"
    }

    for attempt in range(3):
        try:
            r = requests.get(url, params=params)
            if r.status_code == 200:
                json_data = r.json()
                if "Time Series (Daily)" in json_data:
                    data = pd.DataFrame(json_data["Time Series (Daily)"]).T
                    data.columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividend", "Split"]
                    data = data[["Open", "High", "Low", "Close", "Volume"]]
                    data = data.astype(float)
                    data.index = pd.to_datetime(data.index)
                    return data.sort_index()
        except Exception as e:
            print(f"[AlphaVantage] Attempt {attempt+1} failed: {e}")
        time.sleep(3)

    return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
