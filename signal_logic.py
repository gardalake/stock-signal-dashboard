## signal_logic.py - v1.6.5
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
    
    # In un sistema reale, potresti aggiungere colonne come 'buy_price_with_spread' o 'sell_price_with_spread'.
    # df['buy_price_with_spread'] = df['Close'] * (1 + spread_value) # Esempio per acquisto
    # df['sell_price_with_spread'] = df['Close'] * (1 - spread_value) # Esempio per vendita
    # Ma questo dipende da come vuoi usare i segnali. Per ora, non modifichiamo 'Close'.
    
    return df


def combine_signals(
    df_ml_signals: pd.DataFrame, # DataFrame con colonna 'ml_signal'
    df_breakout_signals: pd.DataFrame # DataFrame con colonna 'breakout_signal'
) -> pd.DataFrame:
    """
    Combina i segnali ML e i segnali di breakout in un DataFrame finale.
    Assicura che gli indici siano allineati.
    """
    st.write("DEBUG [signal_logic]: Combinazione segnali ML e breakout.")
    
    # Assicurati che entrambi i DataFrame abbiano le colonne necessarie
    if 'ml_signal' not in df_ml_signals.columns:
        st.error("[signal_logic] ERRORE: 'ml_signal' non trovato nel DataFrame dei segnali ML.")
        # Potrebbe essere necessario creare una colonna di default o gestire l'errore diversamente
        df_ml_signals['ml_signal'] = "ERROR_NO_ML_SIGNAL_COL"

    if 'breakout_signal' not in df_breakout_signals.columns:
        st.error("[signal_logic] ERRORE: 'breakout_signal' non trovato nel DataFrame dei segnali breakout.")
        df_breakout_signals['breakout_signal'] = "ERROR_NO_BREAKOUT_COL"

    # Unisci usando l'indice. È cruciale che l'indice (Date) sia lo stesso.
    # Se i DataFrame hanno colonne diverse oltre ai segnali (es. Close, MA etc.),
    # potremmo voler unire solo le colonne dei segnali a un DataFrame base.
    # Per ora, assumiamo che il df_ml_signals sia il "principale" e aggiungiamo il breakout_signal.
    df_combined = df_ml_signals.copy()
    df_combined = pd.merge(
        df_combined, 
        df_breakout_signals[['breakout_signal']], # Seleziona solo la colonna del segnale breakout
        left_index=True, 
        right_index=True, 
        how='left' # Mantieni tutti i segnali ML, aggiungi breakout dove disponibile
    )
    
    # Riempi i NaN in 'breakout_signal' che potrebbero derivare dal merge se gli indici non corrispondono perfettamente
    # o se breakout_signal non è disponibile per tutte le date.
    df_combined['breakout_signal'].fillna("NONE", inplace=True)
    
    # Logica per un segnale finale combinato (opzionale, per ora teniamo separati ml_signal e breakout_signal)
    # Esempio: df_combined['final_signal'] = df_combined['ml_signal'] (poi sovrascrivi con logica più complessa)

    st.write("DEBUG [signal_logic]: Segnali ML e breakout combinati.")
    return df_combined


# --- Funzioni per email (Placeholder) ---
def send_signal_email_notification(signal_info: dict, email_config: dict, secrets: dict):
    """
    Placeholder per inviare una notifica email riguardo un segnale.
    `secrets` dovrebbe contenere la password SMTP.
    """
    if not email_config.get("enabled", False):
        # st.info("[signal_logic]: Notifiche email disabilitate in config.yaml.")
        return

    st.write(f"INFO [signal_logic]: Si simulerebbe l'invio email per il segnale: {signal_info} (non implementato).")
    
    # Esempio di come potresti accedere alla password dai secrets:
    # smtp_password = secrets.get(email_config.get("smtp_password_secret_name"))
    # if not smtp_password:
    #     st.error("[signal_logic] ERRORE: Password SMTP non trovata nei secrets per l'invio email.")
    #     return
    
    # Qui ci sarebbe la logica per connettersi al server SMTP (usando smtplib) e inviare l'email.
    # ...

if __name__ == '__main__':
    # Blocco per test standalone
    st.write("--- INIZIO TEST STANDALONE signal_logic.py ---")

    # Creare DataFrame di esempio con predizioni e dati OHLCV
    sample_data = {
        'Close': [100, 102, 101, 103, 105, 104, 106, 108, 107, 110],
        'High':  [101, 103, 102, 104, 106, 105, 107, 109, 108, 111],
        'Low':   [99,  101, 100, 102, 104, 103, 105, 107, 106, 109],
        'Volume':[1000,1500,1200,1800,2000,1700,1900,2200,2100,2500],
        'prediction_3d_pct_change': [-0.01, 0.005, 0.015, -0.002, 0.008, 0.02, -0.005, 0.003, 0.01, 0.001] # Predizioni esempio
    }
    sample_dates = pd.date_range(start='2023-01-01', periods=len(sample_data['Close']), freq='B')
    df_sample_preds = pd.DataFrame(sample_data, index=sample_dates)
    st.write("DataFrame di Esempio con Predizioni:")
    st.dataframe(df_sample_preds)

    # 1. Test generazione segnali ML
    buy_thresh = 0.005  # +0.5%
    sell_thresh = -0.005 # -0.5%
    df_ml_signals_test = generate_signals_from_ml_predictions(
        df_sample_preds,
        prediction_column_name='prediction_3d_pct_change',
        buy_threshold=buy_thresh,
        sell_threshold=sell_thresh
    )
    st.write("\nDataFrame con Segnali ML:")
    st.dataframe(df_ml_signals_test[['Close', 'prediction_3d_pct_change', 'ml_signal']])

    # 2. Test rilevamento breakout
    # Aggiungiamo un breakout fittizio
    df_for_breakout = df_sample_preds.copy()
    # df_for_breakout.loc[df_for_breakout.index[5], 'Close'] = 150 # Forza un breakout bullish
    # df_for_breakout.loc[df_for_breakout.index[5], 'Volume'] = 5000
    
    df_breakout_test = detect_breakout_signals(
        df_for_breakout,
        high_low_period=5, # Periodo più breve per dati di esempio limitati
        volume_avg_factor=1.1,
        volume_period=5
    )
    st.write("\nDataFrame con Segnali Breakout:")
    st.dataframe(df_breakout_test[['Close', 'Volume', '5d_high', '5d_low', 'avg_volume_5d', 'breakout_signal']].tail())

    # 3. Test combinazione segnali
    # Per questo test, df_ml_signals_test è il nostro DataFrame principale che contiene già le colonne OHLCV
    # e la colonna 'ml_signal'. Dobbiamo solo aggiungere 'breakout_signal' da df_breakout_test.
    df_combined_test = combine_signals(df_ml_signals_test, df_breakout_test)
    st.write("\nDataFrame con Segnali Combinati:")
    st.dataframe(df_combined_test[['Close', 'ml_signal', 'breakout_signal']])
    
    # 4. Test applicazione spread (placeholder)
    spread_conf_test = {'stocks': 0.006, 'crypto': 0.025}
    df_with_spread_applied_test = apply_trading_spreads(df_combined_test, "stock", spread_conf_test)
    st.write("\nDataFrame dopo (placeholder) applicazione spread:")
    st.dataframe(df_with_spread_applied_test.head()) # Non dovrebbe cambiare i dati

    # 5. Test notifica email (placeholder)
    email_conf_test = {
        "enabled": True, # Simula abilitato
        "smtp_server": "test_server", "smtp_port": 0, "smtp_user": "test_user",
        "smtp_password_secret_name": "TEST_SMTP_PASS", "recipient_email": "test_recipient"
    }
    # Simula un dizionario di secrets vuoto perché non stiamo testando l'invio reale
    secrets_test = {"TEST_SMTP_PASS": "dummy_password"} 
    if not df_combined_test.empty:
        last_signal_example = df_combined_test.iloc[-1].to_dict()
        last_signal_example['ticker'] = "TEST" # Aggiungi ticker per il test
        st.write(f"\nTest notifica email per l'ultimo segnale: {last_signal_example.get('ml_signal')}")
        send_signal_email_notification(last_signal_example, email_conf_test, secrets_test)
    
    st.write("\n--- FINE TEST STANDALONE signal_logic.py ---")signal_logic.py - version v1.6.5
