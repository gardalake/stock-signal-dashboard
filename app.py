# Version: v1.5.1

import streamlit as st
from signal_logic import generate_signals
from data_utils import get_stock_data
import traceback
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

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

        latest_price = data["Close"].iloc[-1]
        predicted_price = signals["predicted_price"]
        signal = signals["ai_signal"]

        # Delta percentuale
        if predicted_price and latest_price:
            delta_percent = ((predicted_price - latest_price) / latest_price) * 100
            delta_color = "green" if delta_percent > 0 else "red"
            delta_text = f"({delta_percent:+.2f}%)"
        else:
            delta_text = ""
            delta_color = "white"

        # Colore segnale
        signal_color = {"BUY": "green", "SELL": "red", "HOLD": "gray"}.get(signal, "white")

        # Stima tempo previsione
        prediction_time = "in 2 days"  # statico per ora

        # Visualizzazione segnale
        st.markdown(f"""
        <h3 style='color:{signal_color};'>
            Signal: {signal} | <span style='color:white;'>Live Price: {latest_price:.2f}</span> | 
            <span style='color:{delta_color};'>Prediction: {predicted_price:.2f} {delta_text} – {prediction_time}</span>
        </h3>
        """, unsafe_allow_html=True)

        # Candlestick con Plotly
        fig = go.Figure(data=[
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="Candles"
            )
        ])

        # Breakout line
        if "breakout_level" in signals and signals["breakout_level"] is not None:
            fig.add_hline(
                y=signals["breakout_level"],
                line_color="blue",
                line_dash="dash",
                annotation_text="Breakout Level"
            )

        fig.update_layout(
            title=f"{selected_ticker} - Candlestick Chart",
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            template="plotly_white",
            height=600,
            xaxis=dict(
                tickformat="%b %d",
                tickmode="auto",
                nticks=15
            )
        )
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error("❌ Error loading data or generating signal.")
    st.code(traceback.format_exc())

# Legenda visiva
st.markdown("""
### 🧾 Legend
- 🟢 **BUY**: Opportunity to enter position.
- 🔴 **SELL**: Time to exit the position.
- ⚪ **HOLD**: No action suggested.
- 🔷 **Breakout line**: Indicates breakout signal level (support/resistance).
- 📊 **Live Price**: Most recent market price.
- 🔮 **Prediction**: AI forecast with % change and estimated time.
""")

# Error log section
st.divider()
st.subheader("🛠️ Debug Log")
with st.expander("Show Raw Log"):
    st.text_area("Logs", value=traceback.format_exc(), height=200)
