# signal_logic.py - v1.6.5
import pandas as pd
import streamlit as st # Per debug e messaggi

def generate_signals_from_ml_predictions(
    df_with_predictions: pd.DataFrame, 
    prediction_column_name: str, # Es. 'prediction_3d_pct_change'
    buy_threshold: float, 
    sell_threshold: float
) -> pd.DataFrame:
    """
    Genera segnali BUY/SELL/HOLD basati sulla colonna di predizione fornita.
    """
    st.write(f"DEBUG [signal_logic]: Generazione segnali ML dalla colonna '{prediction_column_name}'. Soglie: BUY > {buy_threshold*100:.1f}%, SELL < {sell_threshold*100:.1f}%")
    df = df_with_predictions.copy()
    
    if prediction_column_name not in df.columns:
        st.error(f"[signal_logic] ERRORE: Colonna di predizione '{prediction_column_name}' non trovata nel DataFrame.")
        df['ml_signal'] = "ERROR" # Segnale di errore se la colonna manca
        return df

    def determine_ml_signal(pred_value):
        if pd.isna(pred_value):
            return "HOLD" # O "NONE" se preferisci distinguere da un HOLD attivo
        if pred_value > buy_threshold:
            return "BUY"
        elif pred_value < sell_threshold:
            return "SELL"
        else:
            return "HOLD"

    df['ml_signal'] = df[prediction_column_name].apply(determine_ml_signal)
    st.write(f"DEBUG [signal_logic]: Segnali ML generati. Conteggi: {df['ml_signal'].value_counts().to_dict() if 'ml_signal' in df else 'Colonna ml_signal non creata'}")
    return df


def detect_breakout_signals(
    df_input: pd.DataFrame, 
    high_low_period: int = 20, 
    volume_avg_factor: float = 1.0,
    volume_period: int = 20 # Periodo per calcolare il volume medio, può essere diverso da high_low_period
) -> pd.DataFrame:
    """
    Rileva segnali di breakout (bullish e bearish) basati su massimi/minimi di periodo e volume.
    Bullish: Close > X-day High & Volume > avg_volume * factor
    Bearish: Close < X-day Low & Volume > avg_volume * factor
    """
    st.write(f"DEBUG [signal_logic]: Rilevamento breakout. Periodo High/Low: {high_low_period} giorni. Fattore Volume: {volume_avg_factor}x media di {volume_period} giorni.")
    df = df_input.copy()
    
    required_cols = ['Close', 'High', 'Low', 'Volume']
    if not all(col in df.columns for col in required_cols):
        st.error(f"[signal_logic] ERRORE: Colonne {required_cols} necessarie per il breakout non trovate.")
        df['breakout_signal'] = "ERROR"
        return df

    # Calcola X-day High/Low e media volume. shift(1) per usare i dati fino al giorno prima (evita lookahead bias)
    df[f'{high_low_period}d_high'] = df['High'].rolling(window=high_low_period, min_periods=1).max().shift(1)
    df[f'{high_low_period}d_low'] = df['Low'].rolling(window=high_low_period, min_periods=1).min().shift(1)
    df[f'avg_volume_{volume_period}d'] = df['Volume'].rolling(window=volume_period, min_periods=1).mean().shift(1)

    # Segnali di breakout
    bullish_condition = (df['Close'] > df[f'{high_low_period}d_high']) & \
                        (df['Volume'] > df[f'avg_volume_{volume_period}d'] * volume_avg_factor)
    
    bearish_condition = (df['Close'] < df[f'{high_low_period}d_low']) & \
                        (df['Volume'] > df[f'avg_volume_{volume_period}d'] * volume_avg_factor)

    df['breakout_signal'] = "NONE" # Default
    df.loc[bullish_condition, 'breakout_signal'] = "BULLISH"
    df.loc[bearish_condition, 'breakout_signal'] = "BEARISH"
    
    st.write(f"DEBUG [signal_logic]: Segnali breakout rilevati. Conteggi: {df['breakout_signal'].value_counts().to_dict() if 'breakout_signal' in df else 'Colonna breakout_signal non creata'}")
    return df


def apply_trading_spreads(
    df_with_signals: pd.DataFrame, 
    asset_type: str, 
    spread_config: dict # Es. {'stocks': 0.006, 'crypto': 0.025}
) -> pd.DataFrame:
    """
    Applica gli spread ai prezzi di acquisto/vendita (placeholder).
    Questa funzione attualmente è un placeholder e non modifica i dati,
    ma logga lo spread che verrebbe applicato.
    La vera applicazione dello spread avverrebbe in un backtesting o in un sistema di esecuzione ordini.
    """
    st.write(f"DEBUG [signal_logic]: Applicazione spread per tipo asset '{asset_type}'.")
    df = df_with_signals.copy()
    
    spread_value = 0.0
    if asset_type == "stock" and "stocks" in spread_config:
        spread_value = spread_config["stocks"]
    elif asset_type == "crypto" and "crypto" in spread_config:
        spread_value = spread_config["crypto"]
    else:
        st.warning(f"[signal_logic] ATTENZIONE: Tipo asset '{asset_type}' non riconosciuto per spread o non configurato.")
        return df # Nessuno spread applicato, ritorna il DataFrame originale

    st.write(f"DEBUG [signal_logic]: Spread teorico da considerare per '{asset_type}': {spread_value*100:.2f}%")
    
    return df


def combine_signals(
    df_ml_signals: pd.DataFrame, 
    df_breakout_signals: pd.DataFrame 
) -> pd.DataFrame:
    """
    Combina i segnali ML e i segnali di breakout in un DataFrame finale.
    Assicura che gli indici siano allineati.
    """
    st.write("DEBUG [signal_logic]: Combinazione segnali ML e breakout.")
    
    if 'ml_signal' not in df_ml_signals.columns:
        st.error("[signal_logic] ERRORE: 'ml_signal' non trovato nel DataFrame dei segnali ML.")
        df_ml_signals['ml_signal'] = "ERROR_NO_ML_SIGNAL_COL"

    if 'breakout_signal' not in df_breakout_signals.columns:
        st.error("[signal_logic] ERRORE: 'breakout_signal' non trovato nel DataFrame dei segnali breakout.")
        df_breakout_signals['breakout_signal'] = "ERROR_NO_BREAKOUT_COL"

    df_combined = df_ml_signals.copy()
    if 'breakout_signal' in df_breakout_signals:
        df_combined = pd.merge(
            df_combined, 
            df_breakout_signals[['breakout_signal']], 
            left_index=True, 
            right_index=True, 
            how='left' 
        )
    else: 
        df_combined['breakout_signal'] = "ERROR_BREAKOUT_COL_MISSING_IN_MERGE"

    # Modifica per FutureWarning:
    df_combined['breakout_signal'] = df_combined['breakout_signal'].fillna("NONE")
    
    st.write("DEBUG [signal_logic]: Segnali ML e breakout combinati.")
    return df_combined


def send_signal_email_notification(signal_info: dict, email_config: dict, secrets: dict):
    """
    Placeholder per inviare una notifica email riguardo un segnale.
    `secrets` dovrebbe contenere la password SMTP.
    """
    if not email_config.get("enabled", False):
        return

    st.write(f"INFO [signal_logic]: Si simulerebbe l'invio email per il segnale: {signal_info} (non implementato).")
    
    # smtp_password_secret_key = email_config.get("smtp_password_secret_name")
    # smtp_password = None
    # if smtp_password_secret_key:
    #    smtp_password = secrets.get(smtp_password_secret_key)
    # if not smtp_password:
    #     st.error("[signal_logic] ERRORE: Password SMTP non trovata nei secrets per l'invio email.")
    #     return
    

if __name__ == '__main__':
    st.write("--- INIZIO TEST STANDALONE signal_logic.py ---")

    sample_data = {
        'Close': [100, 102, 101, 103, 105, 104, 106, 108, 107, 110],
        'High':  [101, 103, 102, 104, 106, 105, 107, 109, 108, 111],
        'Low':   [99,  101, 100, 102, 104, 103, 105, 107, 106, 109],
        'Volume':[1000,1500,1200,1800,2000,1700,1900,2200,2100,2500],
        'prediction_3d_pct_change': [-0.01, 0.005, 0.015, -0.002, 0.008, 0.02, -0.005, 0.003, 0.01, 0.001] 
    }
    sample_dates = pd.date_range(start='2023-01-01', periods=len(sample_data['Close']), freq='B')
    df_sample_preds = pd.DataFrame(sample_data, index=sample_dates)
    st.write("DataFrame di Esempio con Predizioni:")
    st.dataframe(df_sample_preds)

    buy_thresh = 0.005  
    sell_thresh = -0.005 
    df_ml_signals_test = generate_signals_from_ml_predictions(
        df_sample_preds,
        prediction_column_name='prediction_3d_pct_change',
        buy_threshold=buy_thresh,
        sell_threshold=sell_thresh
    )
    st.write("\nDataFrame con Segnali ML:")
    st.dataframe(df_ml_signals_test[['Close', 'prediction_3d_pct_change', 'ml_signal']])

    df_for_breakout = df_sample_preds.copy()
    df_breakout_test = detect_breakout_signals(
        df_for_breakout,
        high_low_period=5, 
        volume_avg_factor=1.1,
        volume_period=5
    )
    st.write("\nDataFrame con Segnali Breakout:")
    if 'breakout_signal' in df_breakout_test.columns: 
        st.dataframe(df_breakout_test[['Close', 'Volume', '5d_high', '5d_low', 'avg_volume_5d', 'breakout_signal']].tail())
    else:
        st.warning("Colonna 'breakout_signal' non trovata in df_breakout_test.")

    df_combined_test = combine_signals(df_ml_signals_test, df_breakout_test)
    st.write("\nDataFrame con Segnali Combinati:")
    st.dataframe(df_combined_test[['Close', 'ml_signal', 'breakout_signal']])
    
    spread_conf_test = {'stocks': 0.006, 'crypto': 0.025}
    df_with_spread_applied_test = apply_trading_spreads(df_combined_test, "stock", spread_conf_test)
    st.write("\nDataFrame dopo (placeholder) applicazione spread:")
    st.dataframe(df_with_spread_applied_test.head()) 

    email_conf_test = {
        "enabled": True, 
        "smtp_server": "test_server", "smtp_port": 0, "smtp_user": "test_user",
        "smtp_password_secret_name": "TEST_SMTP_PASS", "recipient_email": "test_recipient"
    }
    secrets_test = {"TEST_SMTP_PASS": "dummy_password"} 
    if not df_combined_test.empty:
        last_signal_example = df_combined_test.iloc[-1].to_dict()
        last_signal_example['ticker'] = "TEST" 
        st.write(f"\nTest notifica email per l'ultimo segnale: {last_signal_example.get('ml_signal')}")
        send_signal_email_notification(last_signal_example, email_conf_test, secrets_test)
    
    st.write("\n--- FINE TEST STANDALONE signal_logic.py ---")
