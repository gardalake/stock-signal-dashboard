# Version: v1.6.2

import streamlit as st
from signal_logic import generate_signals
from data_utils import get_stock_data
import traceback
import plotly.graph_objects as go
import datetime

st.set_page_config(page_title="Stock Signal Dashboard", layout="wide")

st.title("📈 AI-Powered Stock & Crypto Signal Dashboard")

selected_ticker = st.selectbox("Select Ticker", [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA",
    "RGTI", "IONQ", "BTC-USD", "ETH-USD", "SOL-USD", "RNDR-USD"
])

try:
    st.markdown("### Legend")
    st.markdown("- **Buy**: Green, **Sell**: Red, **Hold**: White")
    st.markdown("- Arrows (🔺/🔻) indicate direction, % = expected change")
    st.markdown("- Live price updates every ~15 min. AI predictions: 1h → 30d")

    api_key = st.secrets["ALPHA_VANTAGE_API_KEY"]
    data = get_stock_data(selected_ticker, api_key)
    if data is None or data.empty:
        st.warning("⚠️ No data available for this ticker.")
    else:
        signals = generate_signals(data, selected_ticker)

        signal = signals["ai_signal"]
        color = {"BUY": "green", "SELL": "red", "HOLD": "white"}.get(signal, "white")

        st.subheader(f"Signal: :{color}[{signal}] | Live Price: {signals['live_price']}")

        pred = signals["predictions"]
        st.markdown("#### AI Predictions:")
        cols = st.columns(7)
        timeframes = ["1h", "1d", "3d", "5d", "7d", "14d", "30d"]
        for i, tf in enumerate(timeframes):
            with cols[i]:
                st.metric(label=tf, value=pred.get(tf, "N/A"))

        # Crea grafico con Plotly
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data["Open"], high=data["High"],
            low=data["Low"], close=data["Close"],
            name="Candles"
        ))
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error("❌ Error loading data or generating signal.")
    st.exception(e)
