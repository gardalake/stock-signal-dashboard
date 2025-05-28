# Version: v1.4.0

import pandas as pd
import requests
import time

def get_stock_data(symbol, api_key):
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "apikey": api_key,
        "outputsize": "compact"
    }
    r = requests.get(url, params=params)
    while r.status_code != 200:
        time.sleep(1)
        r = requests.get(url, params=params)

    json_data = r.json()
    if "Time Series (Daily)" not in json_data:
        raise ValueError("API response does not contain 'Time Series (Daily)'")

    data = pd.DataFrame(json_data["Time Series (Daily)"]).T
    data.columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividend", "Split"]
    data = data[["Open", "High", "Low", "Close", "Volume"]]
    data = data.astype(float)
    data.index = pd.to_datetime(data.index)
    return data.sort_index()
