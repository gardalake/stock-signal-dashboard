# signal_logic.py - v1.6.5 (Python Logger integration)
import pandas as pd
import streamlit as st # Mantenuto per st.error/st.warning/st.info diretti all'utente

# Importa il setup del logger
from logger_utils import setup_logger
logger = setup_logger(__name__) # Configura un logger per questo modulo

def generate_signals_from_ml_predictions(
    df_with_predictions: pd.DataFrame, 
    prediction_column_name: str, 
    buy_threshold: float, 
    sell_threshold: float
) -> pd.DataFrame:
    logger.debug(f"Generazione segnali ML dalla colonna '{prediction_column_name}'. Soglie: BUY > {buy_threshold*100:.1f}%, SELL < {sell_threshold*100:.1f}%")
    df = df_with_predictions.copy()
    
    if prediction_column_name not in df.columns:
        st.error(f"[SIGNAL_LOGIC] ERRORE: Colonna di predizione '{prediction_column_name}' non trovata.")
        logger.error(f"Colonna di predizione '{prediction_column_name}' non trovata nel DataFrame.")
        df['ml_signal'] = "ERROR_PRED_COL_MISSING" 
        return df

    def determine_ml_signal(pred_value):
        if pd.isna(pred_value):
            return "HOLD" 
        if pred_value > buy_threshold:
            return "BUY"
        elif pred_value < sell_threshold:
            return "SELL"
        else:
            return "HOLD"

    df['ml_signal'] = df[prediction_column_name].apply(determine_ml_signal)
    counts_dict = df['ml_signal'].value_counts().to_dict() if 'ml_signal' in df else 'Colonna ml_signal non creata'
    logger.debug(f"Segnali ML generati. Conteggi: {counts_dict}")
    return df


def detect_breakout_signals(
    df_input: pd.DataFrame, 
    high_low_period: int = 20, 
    volume_avg_factor: float = 1.0,
    volume_period: int = 20 
) -> pd.DataFrame:
    logger.debug(f"Rilevamento breakout. Periodo H/L: {high_low_period}d. Fattore Vol: {volume_avg_factor}x media {volume_period}d.")
    df = df_input.copy()
    
    required_cols = ['Close', 'High', 'Low', 'Volume']
    if not all(col in df.columns for col in required_cols):
        st.error(f"[SIGNAL_LOGIC] ERRORE: Colonne {required_cols} necessarie per breakout non trovate.")
        logger.error(f"Colonne {required_cols} necessarie per breakout non trovate.")
        df['breakout_signal'] = "ERROR_OHLCV_MISSING"
        return df

    # Usare min_periods=1 per evitare NaN all'inizio se il df è più corto del periodo
    # ma questo potrebbe dare breakout "falsi" se i dati sono troppo pochi.
    # È meglio assicurarsi che ci siano abbastanza dati a monte.
    df[f'{high_low_period}d_high'] = df['High'].rolling(window=high_low_period, min_periods=high_low_period//2).max().shift(1)
    df[f'{high_low_period}d_low'] = df['Low'].rolling(window=high_low_period, min_periods=high_low_period//2).min().shift(1)
    df[f'avg_volume_{volume_period}d'] = df['Volume'].rolling(window=volume_period, min_periods=volume_period//2).mean().shift(1)

    bullish_condition = (df['Close'] > df[f'{high_low_period}d_high']) & \
                        (df['Volume'] > df[f'avg_volume_{volume_period}d'] * volume_avg_factor)
    
    bearish_condition = (df['Close'] < df[f'{high_low_period}d_low']) & \
                        (df['Volume'] > df[f'avg_volume_{volume_period}d'] * volume_avg_factor)

    df['breakout_signal'] = "NONE" 
    df.loc[bullish_condition, 'breakout_signal'] = "BULLISH"
    df.loc[bearish_condition, 'breakout_signal'] = "BEARISH"
    
    counts_dict = df['breakout_signal'].value_counts().to_dict() if 'breakout_signal' in df else 'Colonna breakout_signal non creata'
    logger.debug(f"Segnali breakout rilevati. Conteggi: {counts_dict}")
    return df


def apply_trading_spreads(
    df_with_signals: pd.DataFrame, 
    asset_type: str, 
    spread_config: dict 
) -> pd.DataFrame:
    logger.debug(f"Applicazione spread per tipo asset '{asset_type}'.")
    df = df_with_signals.copy()
    
    spread_value = 0.0
    if asset_type == "stock" and "stocks" in spread_config:
        spread_value = spread_config["stocks"]
    elif asset_type == "crypto" and "crypto" in spread_config:
        spread_value = spread_config["crypto"]
    else:
        st.warning(f"[SIGNAL_LOGIC] ATTENZIONE: Tipo asset '{asset_type}' non riconosciuto per spread o non configurato.")
        logger.warning(f"Tipo asset '{asset_type}' non riconosciuto per spread o non configurato.")
        return df 

    logger.debug(f"Spread teorico da considerare per '{asset_type}': {spread_value*100:.2f}%")
    return df


def combine_signals(
    df_ml_signals: pd.DataFrame, 
    df_breakout_signals: pd.DataFrame 
) -> pd.DataFrame:
    logger.debug("Combinazione segnali ML e breakout.")
    
    if 'ml_signal' not in df_ml_signals.columns:
        st.error("[SIGNAL_LOGIC] ERRORE: 'ml_signal' non trovato nel DataFrame dei segnali ML.")
        logger.error("'ml_signal' non trovato nel DataFrame dei segnali ML.")
        df_ml_signals['ml_signal'] = "ERROR_NO_ML_SIGNAL_COL"

    # Non è un errore se df_breakout_signals non ha la colonna, ma va gestito.
    if 'breakout_signal' not in df_breakout_signals.columns:
        logger.warning("'breakout_signal' non trovato nel DataFrame dei segnali breakout. Verrà creata una colonna di default.")
        # Crea una colonna 'breakout_signal' di default se manca, per evitare errori nel merge
        df_breakout_signals_safe = df_breakout_signals.copy()
        df_breakout_signals_safe['breakout_signal'] = "NONE" # O "ERROR_NO_BREAKOUT_COL"
    else:
        df_breakout_signals_safe = df_breakout_signals[['breakout_signal']]


    df_combined = df_ml_signals.copy()
    df_combined = pd.merge(
        df_combined, 
        df_breakout_signals_safe, # Usa la versione sicura
        left_index=True, 
        right_index=True, 
        how='left' 
    )
    
    df_combined['breakout_signal'] = df_combined['breakout_signal'].fillna("NONE")
    
    logger.debug("Segnali ML e breakout combinati.")
    return df_combined


def send_signal_email_notification(signal_info: dict, email_config: dict, secrets: dict):
    if not email_config.get("enabled", False):
        return

    # Questo messaggio può essere INFO se è un'operazione normale, o DEBUG se è solo per tracciamento
    st.info(f"[SIGNAL_LOGIC] SIMULAZIONE: Invio email per segnale: {signal_info} (non implementato).")
    logger.info(f"Simulazione invio email per segnale: {signal_info}")
    
    # smtp_password_secret_key = email_config.get("smtp_password_secret_name")
    # smtp_password = None
    # if smtp_password_secret_key:
    #    smtp_password = secrets.get(smtp_password_secret_key)
    # if not smtp_password:
    #     st.error("[SIGNAL_LOGIC] ERRORE: Password SMTP non trovata nei secrets per l'invio email.")
    #     logger.error("Password SMTP non trovata nei secrets per invio email.")
    #     return
    

if __name__ == '__main__':
    if 'logger' not in locals():
        import logging
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout) # sys non importato qui
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)

    logger.info("--- INIZIO TEST STANDALONE signal_logic.py ---")

    sample_data = {
        'Close': [100, 102, 101, 103, 105, 104, 106, 108, 107, 110],
        'High':  [101, 103, 102, 104, 106, 105, 107, 109, 108, 111],
        'Low':   [99,  101, 100, 102, 104, 103, 105, 107, 106, 109],
        'Volume':[1000,1500,1200,1800,2000,1700,1900,2200,2100,2500],
        'prediction_3d_pct_change': [-0.01, 0.005, 0.015, -0.002, 0.008, 0.02, -0.005, 0.003, 0.01, 0.001] 
    }
    sample_dates = pd.date_range(start='2023-01-01', periods=len(sample_data['Close']), freq='B')
    df_sample_preds = pd.DataFrame(sample_data, index=sample_dates)
    logger.info("DataFrame di Esempio con Predizioni:\n" + df_sample_preds.to_string())

    buy_thresh = 0.005  
    sell_thresh = -0.005 
    df_ml_signals_test = generate_signals_from_ml_predictions(
        df_sample_preds,
        prediction_column_name='prediction_3d_pct_change',
        buy_threshold=buy_thresh,
        sell_threshold=sell_thresh
    )
    logger.info("\nDataFrame con Segnali ML:\n" + df_ml_signals_test[['Close', 'prediction_3d_pct_change', 'ml_signal']].to_string())

    df_for_breakout = df_sample_preds.copy()
    df_breakout_test = detect_breakout_signals(
        df_for_breakout,
        high_low_period=5, 
        volume_avg_factor=1.1,
        volume_period=5
    )
    logger.info("\nDataFrame con Segnali Breakout:")
    if 'breakout_signal' in df_breakout_test.columns: 
        logger.info("\n" + df_breakout_test[['Close', 'Volume', '5d_high', '5d_low', 'avg_volume_5d', 'breakout_signal']].tail().to_string())
    else:
        logger.warning("Colonna 'breakout_signal' non trovata in df_breakout_test.")

    df_combined_test = combine_signals(df_ml_signals_test, df_breakout_test)
    logger.info("\nDataFrame con Segnali Combinati:\n" + df_combined_test[['Close', 'ml_signal', 'breakout_signal']].to_string())
    
    spread_conf_test = {'stocks': 0.006, 'crypto': 0.025}
    df_with_spread_applied_test = apply_trading_spreads(df_combined_test, "stock", spread_conf_test)
    logger.info("\nDataFrame dopo (placeholder) applicazione spread:\n" + df_with_spread_applied_test.head().to_string())

    email_conf_test = {
        "enabled": True, 
        "smtp_server": "test_server", "smtp_port": 0, "smtp_user": "test_user",
        "smtp_password_secret_name": "TEST_SMTP_PASS", "recipient_email": "test_recipient"
    }
    secrets_test = {"TEST_SMTP_PASS": "dummy_password"} 
    if not df_combined_test.empty:
        last_signal_example = df_combined_test.iloc[-1].to_dict()
        last_signal_example['ticker'] = "TEST" 
        logger.info(f"\nTest notifica email per l'ultimo segnale: {last_signal_example.get('ml_signal')}")
        send_signal_email_notification(last_signal_example, email_conf_test, secrets_test)
    
    logger.info("\n--- FINE TEST STANDALONE signal_logic.py ---")
