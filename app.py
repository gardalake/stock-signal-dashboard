# Version: v1.5.0

import streamlit as st
from signal_logic import generate_signals
from data_utils import get_stock_data
import traceback
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os

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

        # Candlestick Plotly
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Candles"
        )])

        if "breakout_level" in signals:
            fig.add_hline(y=signals["breakout_level"], line_color="blue", line_dash="dash", annotation_text="Breakout Level")

        fig.update_layout(
            title=f"{selected_ticker} - Candlestick Chart",
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            template="plotly_white",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

        # Save to signal history
        hist_row = pd.DataFrame([{
            "timestamp": datetime.now(),
            "ticker": selected_ticker,
            "signal": signals['ai_signal'],
            "prediction": signals['predicted_price']
        }])
        if os.path.exists("signal_history.csv"):
            hist_row.to_csv("signal_history.csv", mode="a", header=False, index=False)
        else:
            hist_row.to_csv("signal_history.csv", index=False)

        # Display history
        if os.path.exists("signal_history.csv"):
            st.subheader("📜 Signal History")
            df_hist = pd.read_csv("signal_history.csv")
            st.dataframe(df_hist.tail(20))

except Exception as e:
    st.error("❌ Error loading data or generating signal.")
    st.code(traceback.format_exc())

st.divider()
st.subheader("🛠️ Debug Log")
with st.expander("Show Raw Log"):
    st.text_area("Logs", value=traceback.format_exc(), height=200)
