# Version: v1.6.0

import streamlit as st
from signal_logic import generate_signals
from data_utils import get_stock_data
import traceback
import pandas as pd
import plotly.graph_objects as go

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
        latest_price = signals["live_price"]
        predictions = signals["predictions"]
        signal = signals["ai_signal"]

        # Colori
        signal_color = {"BUY": "green", "SELL": "red", "HOLD": "gray"}.get(signal, "white")

        # Visualizzazione previsioni
        def format_prediction(days, pred):
            if pd.isna(pred):
                return f"{days}d: N/A"
            pct = ((pred - latest_price) / latest_price) * 100
            arrow = "🔺" if pct > 0 else "🔻"
            color = "green" if pct > 0 else "red"
            return f"<span style='color:{color};'>{days}d: {pred:.2f} {arrow} ({pct:+.2f}%)</span>"

        pred_text = " | ".join([format_prediction(f"{h}", predictions.get(f"{h}d")) for h in [1, 3, 5, 7]])

        # Visualizzazione segnale principale
        st.markdown(f"""
        <h3 style='color:{signal_color};'>
            Signal: {signal} | <span style='color:white;'>Live Price: {latest_price:.2f}</span>
        </h3>
        <p style='font-size:18px;'>{pred_text}</p>
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

st.markdown("""
### 🧾 Legend
- 🟢 **BUY**: Opportunity to enter position.
- 🔴 **SELL**: Time to exit the position.
- ⚪ **HOLD**: No action suggested.
- 🔷 **Breakout line**: Indicates breakout signal level (support/resistance).
- 📊 **Live Price**: Most recent market price (updates every 15 min).
- 🔮 **Prediction**: AI forecast at 1, 3, 5, 7 days with % change.
""")

st.divider()
st.subheader("🛠️ Debug Log")
with st.expander("Show Raw Log"):
    st.text_area("Logs", value=traceback.format_exc(), height=200)
