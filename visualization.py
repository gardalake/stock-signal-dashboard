# visualization.py - v1.6.5
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st # Per debug e messaggi

def create_main_stock_chart(
    df_ohlcv_ma: pd.DataFrame, # DataFrame con OHLCV e colonne MA (es. MA20, MA50)
    df_signals: pd.DataFrame,  # DataFrame con colonne 'ml_signal' e 'breakout_signal' (e idealmente 'Close' per posizionare i marker)
    ticker: str,
    ma_periods_to_show: list | None = None, # Es. [20, 50]
    # show_rsi: bool = False, # TODO: Implementare subplot RSI
    # show_stoch_rsi: bool = False # TODO: Implementare subplot StochRSI
) -> go.Figure:
    """
    Crea il grafico principale a candele con Plotly, includendo:
    - Candele OHLC
    - Medie Mobili (se specificate e presenti in df_ohlcv_ma)
    - Segnali ML (BUY/SELL) da df_signals
    - Segnali di Breakout da df_signals
    """
    st.write(f"DEBUG [visualization]: Creazione grafico per {ticker}.")
    
    if df_ohlcv_ma.empty or not all(col in df_ohlcv_ma.columns for col in ['Open', 'High', 'Low', 'Close']):
        st.error("[visualization] ERRORE: Dati OHLC mancanti o insufficienti per creare il grafico.")
        fig = go.Figure() # Figura vuota
        fig.update_layout(title_text=f"Dati non disponibili per {ticker}")
        return fig

    # Determina il numero di subplot necessari. Per ora, 1 riga, 1 colonna.
    # Se aggiungiamo RSI/StochRSI, questo cambierà.
    fig = make_subplots(
        rows=1, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        # subplot_titles=(f"{ticker} Price Action", "RSI", "StochRSI") # Se avessimo 3 righe
        row_heights=[1.0] # Ripartizione altezza righe
    )

    # 1. Grafico a Candele
    fig.add_trace(
        go.Candlestick(
            x=df_ohlcv_ma.index,
            open=df_ohlcv_ma['Open'],
            high=df_ohlcv_ma['High'],
            low=df_ohlcv_ma['Low'],
            close=df_ohlcv_ma['Close'],
            name=f'{ticker} OHLC'
        ),
        row=1, col=1
    )

    # 2. Medie Mobili
    if ma_periods_to_show:
        for period in ma_periods_to_show:
            ma_col_name = f'MA{period}'
            if ma_col_name in df_ohlcv_ma.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_ohlcv_ma.index, 
                        y=df_ohlcv_ma[ma_col_name], 
                        mode='lines', 
                        name=ma_col_name,
                        line=dict(width=1.5) # Linea leggermente più spessa
                    ),
                    row=1, col=1
                )
            else:
                st.warning(f"[visualization] ATTENZIONE: Colonna media mobile '{ma_col_name}' non trovata nel DataFrame per il plot.")

    # 3. Segnali ML (BUY/SELL)
    # Assicurati che df_signals abbia un indice Date e una colonna 'Close' per posizionare i marker.
    # Se df_signals non ha 'Close', dovremmo fare un merge o passare il 'Close' da df_ohlcv_ma.
    # Per ora, assumiamo che df_signals possa essere usato direttamente o che contenga già 'Close'
    # allineato con il suo indice. Una soluzione robusta sarebbe unire df_signals con df_ohlcv_ma['Close'].
    
    # Uniamo df_signals con la colonna 'Close' di df_ohlcv_ma per posizionare correttamente i marker
    # Questo è più sicuro che assumere che df_signals abbia già 'Close'.
    if not df_signals.empty and 'ml_signal' in df_signals.columns:
        # Merge df_signals (che ha 'ml_signal' e 'breakout_signal') con df_ohlcv_ma['Close'] e df_ohlcv_ma['Low'/'High'] per i breakout
        # per avere il prezzo corretto a cui plottare il segnale.
        # L'indice di df_signals deve corrispondere a quello di df_ohlcv_ma.
        df_plot_signals = pd.merge(df_ohlcv_ma[['Close', 'Low', 'High']], df_signals[['ml_signal', 'breakout_signal']], 
                                   left_index=True, right_index=True, how='inner')


        buy_signals = df_plot_signals[df_plot_signals['ml_signal'] == 'BUY']
        sell_signals = df_plot_signals[df_plot_signals['ml_signal'] == 'SELL']

        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index, 
                    y=buy_signals['Close'], # Marker sul prezzo di chiusura del segnale
                    mode='markers', 
                    name='BUY Signal (ML)',
                    marker=dict(symbol='triangle-up', color='rgba(0, 200, 0, 0.9)', size=10,
                                line=dict(width=1, color='DarkSlateGrey'))
                ),
                row=1, col=1
            )
        
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index, 
                    y=sell_signals['Close'], # Marker sul prezzo di chiusura del segnale
                    mode='markers', 
                    name='SELL Signal (ML)',
                    marker=dict(symbol='triangle-down', color='rgba(255, 0, 0, 0.9)', size=10,
                                line=dict(width=1, color='DarkSlateGrey'))
                ),
                row=1, col=1
            )

        # 4. Segnali di Breakout
        # Usiamo sempre df_plot_signals che ora ha Close, Low, High e i segnali.
        bullish_breakouts = df_plot_signals[df_plot_signals['breakout_signal'] == 'BULLISH']
        bearish_breakouts = df_plot_signals[df_plot_signals['breakout_signal'] == 'BEARISH']

        if not bullish_breakouts.empty:
            fig.add_trace(
                go.Scatter(
                    x=bullish_breakouts.index, 
                    y=bullish_breakouts['Low'] * 0.99, # Posiziona leggermente sotto la candela Low
                    mode='markers', 
                    name='Bullish Breakout',
                    marker=dict(symbol='circle-open', color='rgba(0, 0, 255, 0.7)', size=12, 
                                line=dict(width=2))
                ),
                row=1, col=1
            )
        
        if not bearish_breakouts.empty:
            fig.add_trace(
                go.Scatter(
                    x=bearish_breakouts.index, 
                    y=bearish_breakouts['High'] * 1.01, # Posiziona leggermente sopra la candela High
                    mode='markers', 
                    name='Bearish Breakout',
                    marker=dict(symbol='circle-open', color='rgba(255, 165, 0, 0.7)', size=12,
                                line=dict(width=2)) # Arancione per bearish breakout
                ),
                row=1, col=1
            )
    else:
        st.write("DEBUG [visualization]: Nessun segnale fornito a df_signals o colonna 'ml_signal' mancante.")


    # Layout e Stile del Grafico
    fig.update_layout(
        title_text=f'{ticker} - Analisi Tecnica e Segnali',
        xaxis_title='Data',
        yaxis_title='Prezzo',
        xaxis_rangeslider_visible=False, # Slider sotto il grafico principale
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=600, # Altezza del grafico in pixel
        margin=dict(l=50, r=50, t=80, b=50), # Margini
        # hovermode='x unified' # Mostra tutti i dati per una data quando si passa il mouse
    )
    
    # TODO: Aggiungere subplot per RSI e StochRSI
    # if show_rsi:
    #   ...
    # if show_stoch_rsi:
    #   ...

    st.write("DEBUG [visualization]: Grafico creato con successo.")
    return fig


if __name__ == '__main__':
    # Blocco per test standalone
    st.write("--- INIZIO TEST STANDALONE visualization.py ---")

    # Creare DataFrame di esempio per OHLCV e MA
    sample_ohlcv_data = {
        'Open':  [100, 102, 101, 103, 105, 104, 106, 108, 107, 110],
        'High':  [101, 103, 102, 104, 106, 105, 107, 109, 108, 111],
        'Low':   [99,  101, 100, 102, 104, 103, 105, 107, 106, 109],
        'Close': [100, 102, 101, 103, 105, 104, 106, 108, 107, 110],
        'MA20':  [100, 101, 101, 101.5, 102.4, 103, 103.8, 104.7, 105.4, 106.2], # Valori MA fittizi
        'MA50':  [100, 100, 100, 100.2, 100.5, 100.8, 101.2, 101.7, 102.1, 102.6]  # Valori MA fittizi
    }
    sample_dates_viz = pd.date_range(start='2023-01-01', periods=len(sample_ohlcv_data['Close']), freq='B')
    df_ohlcv_ma_test = pd.DataFrame(sample_ohlcv_data, index=sample_dates_viz)
    st.write("DataFrame OHLCV + MA di Esempio:")
    st.dataframe(df_ohlcv_ma_test)

    # Creare DataFrame di esempio per i segnali
    sample_signals_data = {
        'ml_signal':       ['HOLD', 'BUY',  'HOLD', 'SELL', 'HOLD', 'BUY',  'HOLD', 'HOLD', 'SELL', 'HOLD'],
        'breakout_signal': ['NONE', 'NONE', 'NONE', 'NONE', 'BULLISH','NONE', 'NONE', 'BEARISH','NONE', 'NONE']
    }
    df_signals_test = pd.DataFrame(sample_signals_data, index=sample_dates_viz)
    st.write("\nDataFrame Segnali di Esempio:")
    st.dataframe(df_signals_test)
    
    # Test creazione grafico
    st.write("\n--- Creazione Grafico di Test ---")
    fig_test = create_main_stock_chart(
        df_ohlcv_ma=df_ohlcv_ma_test,
        df_signals=df_signals_test,
        ticker="TESTICKER",
        ma_periods_to_show=[20, 50]
    )
    
    if fig_test:
        st.plotly_chart(fig_test, use_container_width=True)
        st.success("Grafico di test creato e visualizzato.")
    else:
        st.error("Fallita creazione del grafico di test.")
        
    st.write("\n--- FINE TEST STANDALONE visualization.py ---")
