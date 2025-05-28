# Version: v1.6.3

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
    st.markdown("""
        <style>
        .legend-box {
            background-color: #1c1c1e;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            border: 1px solid #333;
        }
        .signal-box {
            background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
            padding: 1.2rem;
            border-radius: 0.75rem;
            margin-top: 1rem;
            margin-bottom: 1rem;
            border-left: 5px solid #0f0;
            color: white;
            font-size: 1.2rem;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="legend-box">
    <strong>📘 Legend</strong><br>
    🔹 <b>Signal:</b> <span style="color:green">BUY</span>, <span style="color:red">SELL</span>, <span style="color:white">HOLD</span><br>
    🔹 <b>Live Price:</b> Updated every 15 minutes.<br>
    🔹 <b>Prediction:</b> Price prediction for 1h, 1d, 3d, 5d, 7d, 14d, and 30d.<br>
    🔹 <b>Trend Arrows:</b> 🔺 (up), 🔻 (down), % = predicted change.
    </div>
    """, unsafe_allow_html=True)

    api_key = st.secrets["ALPHA_VANTAGE_API_KEY"]
    data = get_stock_data(selected_ticker, api_key)
    if data is None or data.empty:
        st.warning("⚠️ No data available for this ticker.")
    else:
        signals = generate_signals(data, selected_ticker)

        signal = signals["ai_signal"]
        color = {"BUY": "green", "SELL": "red", "HOLD": "white"}.get(signal, "white")

        st.markdown(f"<div class='signal-box'>📍 Signal: <b style='color:{color}'>{signal}</b> | Live Price: {signals['live_price']}</div>", unsafe_allow_html=True)

        pred = signals["predictions"]
        st.markdown("#### 🔮 AI Predictions:")
        cols = st.columns(7)
        timeframes = ["1h", "1d", "3d", "5d", "7d", "14d", "30d"]
        for i, tf in enumerate(timeframes):
            with cols[i]:
                st.metric(label=tf, value=pred.get(tf, "N/A"))

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
            height=600,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error("❌ Error loading data or generating signal.")
    st.exception(e)
