# Version: v1.4.0

import streamlit as st
from signal_logic import generate_signals
from data_utils import get_stock_data
import traceback
import mplfinance as mpf
import pandas as pd

st.set_page_config(page_title="AI Stock Signals", layout="wide")

# Load API key and tickers
try:
    api_key = st.secrets["alpha_vantage_api_key"]
    tickers = [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA",
        "RGTI", "IONQ", "BTC-USD", "ETH-USD", "SOL-USD", "RNDR-USD"
    ]
except Exception as e:
    st.error("Missing 'alpha_vantage_api_key' in Streamlit secrets.")
    st.stop()

st.title("📈 AI-Powered Stock Signals")
st.markdown("Candlestick chart with AI, RSI, and breakout signals. Tailored for eToro and crypto spread.")

error_logs = []

try:
    selected_ticker = st.selectbox("Choose a stock or crypto", tickers)
    data = get_stock_data(selected_ticker, api_key)
    signals = generate_signals(data, selected_ticker)

    st.subheader(f"📊 Candlestick Chart for {selected_ticker}")

    ohlc_data = data.copy()
    ohlc_data["Open"] = ohlc_data["Open"].astype(float)
    ohlc_data["High"] = ohlc_data["High"].astype(float)
    ohlc_data["Low"] = ohlc_data["Low"].astype(float)
    ohlc_data["Close"] = ohlc_data["Close"].astype(float)

    buy_signals = [s for s in signals if "BUY" in s["signal"]]
    sell_signals = [s for s in signals if "SELL" in s["signal"]]
    breakout_signals = [s for s in signals if "BREAKOUT" in s["signal"]]

    apds = []
    if buy_signals:
        buy_df = pd.DataFrame(buy_signals).set_index("date")
        apds.append(mpf.make_addplot(buy_df["price"], type="scatter", marker="^", markersize=100, color="green"))
    if sell_signals:
        sell_df = pd.DataFrame(sell_signals).set_index("date")
        apds.append(mpf.make_addplot(sell_df["price"], type="scatter", marker="v", markersize=100, color="red"))
    if breakout_signals:
        bo_df = pd.DataFrame(breakout_signals).set_index("date")
        apds.append(mpf.make_addplot(bo_df["price"], type="scatter", marker="o", markersize=100, color="blue"))

    fig, _ = mpf.plot(
        ohlc_data,
        type="candle",
        style="yahoo",
        addplot=apds,
        returnfig=True,
        volume=True,
        figsize=(12, 6)
    )
    st.pyplot(fig)

    if buy_signals:
        st.audio("https://www.soundjay.com/buttons/sounds/button-4.mp3", format="audio/mp3")
    elif sell_signals:
        st.audio("https://www.soundjay.com/buttons/sounds/beep-07.mp3", format="audio/mp3")

    st.subheader("📜 Full Signal History")
    df_signals = pd.DataFrame(signals)
    df_signals = df_signals.sort_values("date", ascending=False)
    st.dataframe(df_signals)

except Exception as e:
    error_message = traceback.format_exc()
    error_logs.append(error_message)
    st.error("⚠️ An error occurred. Check error logs below.")

if error_logs:
    st.subheader("🛠️ Error Logs")
    st.text_area("Copy and send this to the developer:", value=error_logs[0], height=300)
