# Version: v1.4.2

import streamlit as st
from signal_logic import generate_signals
from data_utils import get_stock_data
import traceback
import mplfinance as mpf
import pandas as pd

# Titoli monitorati
stock_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "RGTI", "IONQ", "BTC-USD", "ETH-USD", "SOL-USD", "RNDR-USD"]

st.set_page_config(layout="wide", page_title="AI Stock Signal Dashboard")

st.title("📈 AI-Powered Stock & Crypto Signal Dashboard")

selected_ticker = st.selectbox("Select Stock or Crypto", stock_list)
api_key = st.secrets["ALPHA_VANTAGE_API_KEY"]

try:
    data = get_stock_data(selected_ticker, api_key)
    if data.empty:
        st.warning("⚠️ No data available for this ticker.")
    else:
        signals = generate_signals(data, selected_ticker)

        st.subheader(f"Signal: {signals['ai_signal']} | Prediction: {signals['predicted_price']:.2f}")

        ohlc_data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        ohlc_data.index = pd.to_datetime(ohlc_data.index)

        fig, _ = mpf.plot(
            ohlc_data,
            type='candle',
            style='charles',
            volume=True,
            returnfig=True,
            figsize=(12, 6),
            title=selected_ticker,
            warn_too_much_data=10000
        )
        st.pyplot(fig)

except Exception as e:
    st.error("❌ Error loading data or generating signal.")
    st.code(traceback.format_exc())

# Error log section
st.divider()
st.subheader("🛠️ Debug Log")
with st.expander("Show Raw Log"):
    st.text_area("Logs", value=traceback.format_exc(), height=200)
