# app.py - v1.6.5 (Main Application Orchestrator - Python Logger, Index Alignment)
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import yaml 
import os 
import time 
import json 

# Importa il setup del logger PRIMA degli altri moduli del progetto
# così se quei moduli usano il logger, è già configurato.
from logger_utils import setup_logger
logger = setup_logger(__name__) # Logger per app.py

# Importa dai moduli del progetto
from data_utils import get_stock_data, get_crypto_data
from ml_model import calculate_technical_features, create_prediction_targets, train_random_forest_model, generate_model_predictions, get_predictions_from_ai_studio
from signal_logic import generate_signals_from_ml_predictions, detect_breakout_signals, apply_trading_spreads, combine_signals, send_signal_email_notification
from visualization import create_main_stock_chart
from sound_utils import play_buy_signal_sound, play_sell_signal_sound

# --- CARICAMENTO CONFIGURAZIONE ---
CONFIG_FILE = "config.yaml"
CONFIG = {}
APP_VERSION_FROM_CONFIG = "N/A" 
config_loaded_successfully_flag = False 
yaml_error_message_for_later = None 

try:
    with open(CONFIG_FILE, 'r') as f:
        CONFIG = yaml.safe_load(f)
    APP_VERSION_FROM_CONFIG = CONFIG.get('version', 'v1.6.5-spec-impl (config fallback)')
    config_loaded_successfully_flag = True
except FileNotFoundError:
    APP_VERSION_FROM_CONFIG = "ERRORE CONFIG - File non trovato" 
    # logger non è ancora garantito che sia configurato al 100% qui se logger_utils fallisce l'import
    # quindi usiamo print per questo errore critico a livello di modulo
    print(f"CRITICAL_ERROR [app.py_module]: {CONFIG_FILE} non trovato.")
except yaml.YAMLError as e:
    APP_VERSION_FROM_CONFIG = "ERRORE CONFIG - YAML invalido"
    yaml_error_message_for_later = e 
    print(f"CRITICAL_ERROR [app.py_module]: Errore parsing {CONFIG_FILE}: {e}")

# --- CONFIGURAZIONE PAGINA STREAMLIT ---
st.set_page_config(
    layout="wide",
    page_title=f"Stock Signal Dashboard {APP_VERSION_FROM_CONFIG}", 
    page_icon="📈" 
)

# Ora che st.set_page_config è stato chiamato, possiamo usare st.error e il logger
if config_loaded_successfully_flag:
    logger.info(f"{CONFIG_FILE} caricato. Versione da config: {APP_VERSION_FROM_CONFIG}")
    if 'config_loaded_successfully' not in st.session_state: 
        st.session_state.config_loaded_successfully = True
elif yaml_error_message_for_later is not None:
    st.error(f"ERRORE CRITICO nel parsing di '{CONFIG_FILE}': {yaml_error_message_for_later}. L'app non può continuare.")
    logger.critical(f"Errore parsing '{CONFIG_FILE}': {yaml_error_message_for_later}")
    st.session_state.config_loaded_successfully = False
    st.stop()
else: # FileNotFoundError
    st.error(f"ERRORE CRITICO: '{CONFIG_FILE}' non trovato. L'app non può continuare.")
    logger.critical(f"'{CONFIG_FILE}' non trovato.")
    st.session_state.config_loaded_successfully = False
    st.stop()

# --- GESTIONE CHIAVI API ---
ALPHA_VANTAGE_API_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY")
if not ALPHA_VANTAGE_API_KEY and CONFIG.get('alpha_vantage'): 
    st.sidebar.warning("Chiave API Alpha Vantage (ALPHA_VANTAGE_API_KEY) non trovata in st.secrets.")
    logger.warning("Chiave API Alpha Vantage non trovata in st.secrets.")

# ... (resto gestione chiavi API come prima, usando logger per warning/error se appropriato) ...
GOOGLE_AI_STUDIO_URL = None
GOOGLE_AI_STUDIO_TOKEN = None
if CONFIG.get('ml_model', {}).get('use_google_ai_studio', False):
    url_secret_name = CONFIG.get('ml_model', {}).get('google_ai_studio_url_secret_name')
    token_secret_name = CONFIG.get('ml_model', {}).get('google_ai_studio_token_secret_name')
    if url_secret_name and token_secret_name:
        GOOGLE_AI_STUDIO_URL = st.secrets.get(url_secret_name)
        GOOGLE_AI_STUDIO_TOKEN = st.secrets.get(token_secret_name)
        if not GOOGLE_AI_STUDIO_URL or not GOOGLE_AI_STUDIO_TOKEN:
            st.sidebar.warning(f"Google AI Studio: URL ({url_secret_name}) o Token ({token_secret_name}) mancanti.")
            logger.warning(f"Google AI Studio: URL ({url_secret_name}) o Token ({token_secret_name}) mancanti in st.secrets.")
    else:
        st.sidebar.error("Nomi secret Google AI Studio non in config.yaml.")
        logger.error("Nomi secret Google AI Studio non specificati in config.yaml.")

EMAIL_SMTP_PASSWORD = None
if CONFIG.get('email_notifications', {}).get('enabled', False):
    pwd_secret_name = CONFIG.get('email_notifications', {}).get('smtp_password_secret_name')
    if pwd_secret_name:
        EMAIL_SMTP_PASSWORD = st.secrets.get(pwd_secret_name)
        if not EMAIL_SMTP_PASSWORD:
            st.sidebar.warning(f"Email: password SMTP ({pwd_secret_name}) non in st.secrets.")
            logger.warning(f"Email: password SMTP ({pwd_secret_name}) non trovata in st.secrets.")
    else:
        st.sidebar.error("Nome secret password SMTP non in config.yaml.")
        logger.error("Nome secret password SMTP non specificato in config.yaml.")

LOADED_SECRETS = {
    "ALPHA_VANTAGE_API_KEY": ALPHA_VANTAGE_API_KEY,
    CONFIG.get('ml_model', {}).get('google_ai_studio_url_secret_name', 'GOOGLE_AI_STUDIO_URL_PLACEHOLDER'): GOOGLE_AI_STUDIO_URL,
    CONFIG.get('ml_model', {}).get('google_ai_studio_token_secret_name', 'GOOGLE_AI_STUDIO_TOKEN_PLACEHOLDER'): GOOGLE_AI_STUDIO_TOKEN,
    CONFIG.get('email_notifications', {}).get('smtp_password_secret_name', 'EMAIL_SMTP_PASSWORD_PLACEHOLDER'): EMAIL_SMTP_PASSWORD
}

# --- STATO DELLA SESSIONE ---
default_session_state_values = {
    'ss_ticker_input': "AAPL", 
    'ss_asset_type': "stock", 
    'ss_start_date_display': datetime.now().date() - timedelta(days=90), 
    'ss_end_date_display': datetime.now().date(),     
    'ss_raw_data_df_full_history': None, 
    'ss_raw_data_df_filtered': None,   
    'ss_features_df_full_history': None, # Rinominato per chiarezza (contiene feature su full_history)      
    'ss_target_and_preds_df_full_history': None, # Rinominato
    'ss_final_signals_df_display': None, # Rinominato (contiene segnali filtrati per display) 
    'ss_trained_ml_model': None,     
    'ss_last_generated_signal_info': None, 
    'ss_analysis_run_triggered': False 
}
for key, value in default_session_state_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- TITOLO E HEADER ---
st.title(f"📈 Stock & Crypto Signal Dashboard")
st.markdown(f"**Versione:** `{APP_VERSION_FROM_CONFIG}` - _Basato sulle specifiche del progetto._")
st.markdown("---")

# --- SIDEBAR ---
with st.sidebar:
    st.header("🛠️ Controlli e Parametri")
    # ... (contenuto sidebar come prima) ...
    st.session_state.ss_asset_type = st.radio(
        "Seleziona Tipo Asset:",
        options=["stock", "crypto"],
        index=["stock", "crypto"].index(st.session_state.ss_asset_type), 
        horizontal=True,
        key="asset_type_radio" 
    )

    default_ticker_placeholder = "Es. AAPL, TSLA" if st.session_state.ss_asset_type == "stock" else "Es. bitcoin, ethereum"
    if 'prev_asset_type' not in st.session_state: st.session_state.prev_asset_type = st.session_state.ss_asset_type
    if st.session_state.prev_asset_type != st.session_state.ss_asset_type:
        st.session_state.ss_ticker_input = "AAPL" if st.session_state.ss_asset_type == "stock" else "bitcoin" 
    st.session_state.prev_asset_type = st.session_state.ss_asset_type
    current_ticker_value = st.session_state.get('ss_ticker_input', "AAPL" if st.session_state.ss_asset_type == "stock" else "bitcoin")
    
    st.session_state.ss_ticker_input = st.text_input(
        f"Inserisci Simbolo ({'Azioni' if st.session_state.ss_asset_type == 'stock' else 'Crypto'}):",
        value=current_ticker_value,
        placeholder=default_ticker_placeholder,
        key="ticker_text_input"
    )
    if st.session_state.ss_asset_type == "stock":
        st.session_state.ss_ticker_input = st.session_state.ss_ticker_input.upper()
    else:
        st.session_state.ss_ticker_input = st.session_state.ss_ticker_input.lower()

    st.markdown("##### Intervallo Date per Visualizzazione/Analisi:")
    col_date1, col_date2 = st.columns(2)
    with col_date1:
        st.session_state.ss_start_date_display = st.date_input("Data Inizio", value=st.session_state.ss_start_date_display, key="start_date_input_display", help="Data inizio per visualizzazione e analisi finale.")
    with col_date2:
        st.session_state.ss_end_date_display = st.date_input("Data Fine", value=st.session_state.ss_end_date_display, key="end_date_input_display", help="Data fine. Deve essere >= Data Inizio.")

    if st.session_state.ss_start_date_display > st.session_state.ss_end_date_display:
        st.error("Errore: La Data di Inizio deve essere precedente o uguale alla Data di Fine.")
        logger.error(f"Errore input date: start {st.session_state.ss_start_date_display} > end {st.session_state.ss_end_date_display}")
        st.stop() 

    st.markdown("---") 
    if st.button("🚀 Analizza e Genera Segnali", type="primary", use_container_width=True, key="run_analysis_button"):
        if not st.session_state.ss_ticker_input:
            st.warning("Per favore, inserisci un simbolo Ticker o Coin ID valido.")
            logger.warning("Tentativo analisi senza ticker input.")
        else:
            st.session_state.ss_analysis_run_triggered = True 
            # Resetta stati specifici
            st.session_state.ss_raw_data_df_full_history = None 
            st.session_state.ss_raw_data_df_filtered = None   
            st.session_state.ss_features_df_full_history = None      
            st.session_state.ss_target_and_preds_df_full_history = None 
            st.session_state.ss_final_signals_df_display = None   
            st.session_state.ss_trained_ml_model = None     
            st.session_state.ss_last_generated_signal_info = None 
            logger.info(f"Avvio analisi per {st.session_state.ss_ticker_input} ({st.session_state.ss_asset_type}) su richiesta utente.")
            # Non serve st.info qui, la pipeline sotto stamperà i suoi messaggi.
    st.markdown("---")


# --- PIPELINE DI ELABORAZIONE DATI E SEGNALI ---
if st.session_state.ss_analysis_run_triggered:
    log_container = st.container() # Contenitore per i messaggi di log della UI
    # Non resettare ss_analysis_run_triggered qui, ma alla FINE della pipeline
    
    with log_container: # I messaggi st.* dentro questo with appariranno nel container
        st.markdown("### ⚙️ Log di Processo dell'Analisi")
        progress_bar = st.progress(0, text="Inizio analisi...")

        MIN_DAYS_HISTORY_FOR_ML = CONFIG.get('ml_model', {}).get('min_days_for_indicators_and_training', 200)
        actual_data_load_start_date = st.session_state.ss_end_date_display - timedelta(days=MIN_DAYS_HISTORY_FOR_ML -1) 

        progress_bar.progress(10, text=f"Caricamento storico esteso per {st.session_state.ss_ticker_input}...")
        logger.info(f"Inizio caricamento dati per {st.session_state.ss_ticker_input}.")
        if st.session_state.ss_asset_type == "stock":
            if not ALPHA_VANTAGE_API_KEY:
                st.error("Impossibile caricare dati stock: Chiave API Alpha Vantage mancante.")
                logger.error("Chiave API Alpha Vantage mancante per caricamento dati stock.")
                st.session_state.ss_raw_data_df_full_history = None 
            else:
                st.session_state.ss_raw_data_df_full_history = get_stock_data(
                    ALPHA_VANTAGE_API_KEY, 
                    st.session_state.ss_ticker_input,
                    CONFIG.get('alpha_vantage', {}).get('function', 'TIME_SERIES_DAILY_ADJUSTED'),
                    "full" 
                )
        elif st.session_state.ss_asset_type == "crypto":
            days_to_fetch_for_coingecko = (datetime.now().date() - actual_data_load_start_date).days + 1
            if days_to_fetch_for_coingecko <=0: days_to_fetch_for_coingecko = MIN_DAYS_HISTORY_FOR_ML 
            logger.debug(f"Caricamento crypto - Data inizio richiesta API: {actual_data_load_start_date}, Giorni da fetchare: {days_to_fetch_for_coingecko}")
            st.session_state.ss_raw_data_df_full_history = get_crypto_data(
                st.session_state.ss_ticker_input,
                vs_currency=CONFIG.get('coingecko',{}).get('vs_currency', 'usd'),
                days=days_to_fetch_for_coingecko 
            )
        
        if st.session_state.ss_raw_data_df_full_history is not None and not st.session_state.ss_raw_data_df_full_history.empty:
            logger.info(f"Storico esteso caricato. Shape: {st.session_state.ss_raw_data_df_full_history.shape}")
            start_dt_display_filter = pd.to_datetime(st.session_state.ss_start_date_display)
            end_dt_display_filter = pd.to_datetime(st.session_state.ss_end_date_display)
            if not isinstance(st.session_state.ss_raw_data_df_full_history.index, pd.DatetimeIndex):
                st.session_state.ss_raw_data_df_full_history.index = pd.to_datetime(st.session_state.ss_raw_data_df_full_history.index)
            
            st.session_state.ss_raw_data_df_filtered = st.session_state.ss_raw_data_df_full_history[
                (st.session_state.ss_raw_data_df_full_history.index >= start_dt_display_filter) & 
                (st.session_state.ss_raw_data_df_full_history.index <= end_dt_display_filter)
            ].copy() 

            if st.session_state.ss_raw_data_df_filtered.empty:
                st.warning(f"Nessun dato per {st.session_state.ss_ticker_input} trovato nell'intervallo di visualizzazione ({st.session_state.ss_start_date_display.strftime('%Y-%m-%d')} - {st.session_state.ss_end_date_display.strftime('%Y-%m-%d')}). L'analisi ML continuerà sullo storico completo se disponibile.")
                logger.warning(f"Dati filtrati per visualizzazione vuoti per {st.session_state.ss_ticker_input}.")
            else:
                st.success(f"Dati per visualizzazione/analisi ({st.session_state.ss_start_date_display.strftime('%Y-%m-%d')} - {st.session_state.ss_end_date_display.strftime('%Y-%m-%d')}): Shape {st.session_state.ss_raw_data_df_filtered.shape}")
        
        elif st.session_state.ss_raw_data_df_full_history is None: 
            st.error(f"Fallimento nel caricamento dello storico esteso per {st.session_state.ss_ticker_input}.")
            logger.error(f"Fallimento caricamento storico esteso per {st.session_state.ss_ticker_input}.")
        
        # --- INIZIO ELABORAZIONE ML (su ss_raw_data_df_full_history) ---
        if st.session_state.ss_raw_data_df_full_history is not None and not st.session_state.ss_raw_data_df_full_history.empty:
            if len(st.session_state.ss_raw_data_df_full_history) < MIN_DAYS_HISTORY_FOR_ML / 2: 
                st.warning(f"Storico caricato ({len(st.session_state.ss_raw_data_df_full_history)} righe) potrebbe essere insufficiente per analisi ML robusta (minimo richiesto: {MIN_DAYS_HISTORY_FOR_ML}).")
                logger.warning(f"Storico caricato per ML ({len(st.session_state.ss_raw_data_df_full_history)} righe) potrebbe essere insufficiente.")
            
            progress_bar.progress(25, text="Calcolo feature tecniche su storico esteso...")
            st.session_state.ss_features_df_full_history = calculate_technical_features(st.session_state.ss_raw_data_df_full_history)
            
            if st.session_state.ss_features_df_full_history.empty or len(st.session_state.ss_features_df_full_history) < 10: 
                st.error("Fallimento nel calcolo delle feature o dati insufficienti dopo calcolo feature.")
                logger.error("Fallimento calcolo feature o dati post-feature insufficienti.")
            else:
                st.success(f"Feature tecniche calcolate su storico esteso. Shape: {st.session_state.ss_features_df_full_history.shape}")
                logger.info(f"Feature calcolate su storico esteso. Shape: {st.session_state.ss_features_df_full_history.shape}")

                progress_bar.progress(40, text="Creazione target di predizione...")
                pred_horizon = CONFIG.get('ml_model', {}).get('prediction_target_horizon_days', 3)
                df_with_target_full_hist = create_prediction_targets(st.session_state.ss_features_df_full_history, horizon=pred_horizon) 
                target_col_name = f'target_{pred_horizon}d_pct_change'

                feature_cols_for_ml_config = CONFIG.get('ml_model',{}).get('feature_columns_for_training', ['MA20', 'MA50', 'RSI', 'StochRSI_K', 'Momentum_ROC10', 'ADX', 'MACD_line'])
                # Filtra le feature effettivamente presenti nel DataFrame dopo il calcolo (potrebbero mancare se 'ta' fallisce o dati scarsi)
                feature_cols_for_ml = [col for col in feature_cols_for_ml_config if col in df_with_target_full_hist.columns]
                if len(feature_cols_for_ml) < len(feature_cols_for_ml_config):
                    missing_configured_features = set(feature_cols_for_ml_config) - set(feature_cols_for_ml)
                    logger.warning(f"Alcune feature configurate per ML non sono state trovate/calcolate: {missing_configured_features}. Si procede con le feature disponibili: {feature_cols_for_ml}")


                if not feature_cols_for_ml:
                    st.error("Nessuna colonna feature valida (dopo calcolo) trovata per il training/predizione ML.")
                    logger.error("Nessuna colonna feature valida per il training ML.")
                elif target_col_name not in df_with_target_full_hist.columns:
                    st.error(f"Colonna target '{target_col_name}' non creata o mancante per il training ML.")
                    logger.error(f"Colonna target '{target_col_name}' non creata/mancante per training ML.")
                else:
                    predictions_series = None 
                    if CONFIG.get('ml_model', {}).get('use_google_ai_studio', False):
                        progress_bar.progress(55, text="Ottenimento predizioni da Google AI Studio...")
                        logger.info("Tentativo predizioni con Google AI Studio (non implementato).")
                        st.warning("Integrazione Google AI Studio non completamente implementata.")
                        predictions_series = pd.Series(index=df_with_target_full_hist.index, dtype=float) 
                    else: 
                        progress_bar.progress(55, text="Training modello RandomForest...")
                        logger.info(f"Inizio training RandomForest. Feature: {feature_cols_for_ml}")
                        st.session_state.ss_trained_ml_model = train_random_forest_model(
                            df_with_target_full_hist, 
                            feature_columns=feature_cols_for_ml,
                            target_column=target_col_name,
                            n_estimators=CONFIG.get('ml_model',{}).get('random_forest_n_estimators', 100)
                        )
                        if st.session_state.ss_trained_ml_model:
                            st.success("Modello RandomForest addestrato.")
                            logger.info("Modello RandomForest addestrato.")
                            progress_bar.progress(70, text="Generazione predizioni ML...")
                            predictions_series = generate_model_predictions(
                                st.session_state.ss_trained_ml_model,
                                df_with_target_full_hist, 
                                feature_columns=feature_cols_for_ml
                            )
                        else:
                            st.error("Fallimento training modello RandomForest.")
                            logger.error("Fallimento training RandomForest.")
                            
                    if predictions_series is not None:
                        st.session_state.ss_target_and_preds_df_full_history = df_with_target_full_hist.copy()
                        prediction_col_ml_name = f'prediction_{pred_horizon}d_pct_change' # Deve corrispondere a quella usata in signal_logic
                        st.session_state.ss_target_and_preds_df_full_history[prediction_col_ml_name] = predictions_series
                        st.success(f"Predizioni ML generate come '{prediction_col_ml_name}' su storico esteso.")
                        logger.info(f"Predizioni ML generate. Colonna: '{prediction_col_ml_name}'.")
                    else:
                        st.error("Fallimento nella generazione delle predizioni ML.")
                        logger.error("Fallimento generazione predizioni ML.")

                if st.session_state.ss_target_and_preds_df_full_history is not None and prediction_col_ml_name in st.session_state.ss_target_and_preds_df_full_history:
                    progress_bar.progress(85, text="Generazione segnali di trading...")
                    logger.info("Inizio generazione segnali di trading.")
                    df_ml_signals_only_full_hist = generate_signals_from_ml_predictions(
                        st.session_state.ss_target_and_preds_df_full_history,
                        prediction_column_name=prediction_col_ml_name,
                        buy_threshold=CONFIG.get('signal_logic',{}).get('buy_threshold_change', 0.005),
                        sell_threshold=CONFIG.get('signal_logic',{}).get('sell_threshold_change', -0.005)
                    )
                    df_breakout_signals_only_full_hist = detect_breakout_signals(
                        st.session_state.ss_features_df_full_history, 
                        high_low_period=CONFIG.get('signal_logic',{}).get('breakout_days_high_low_period', 20),
                        volume_avg_factor=CONFIG.get('signal_logic',{}).get('breakout_volume_avg_factor', 1.0),
                        volume_period=CONFIG.get('signal_logic',{}).get('breakout_days_high_low_period', 20) 
                    )
                    df_signals_full_history_combined = combine_signals(df_ml_signals_only_full_hist, df_breakout_signals_only_full_hist)
                    df_signals_full_history_combined = apply_trading_spreads(
                        df_signals_full_history_combined,
                        st.session_state.ss_asset_type,
                        CONFIG.get('spreads', {})
                    )
                    
                    # Filtra i segnali finali per l'intervallo di visualizzazione
                    if not df_signals_full_history_combined.empty and st.session_state.ss_raw_data_df_filtered is not None and not st.session_state.ss_raw_data_df_filtered.empty:
                        common_index_display = st.session_state.ss_raw_data_df_filtered.index.intersection(df_signals_full_history_combined.index)
                        if not common_index_display.empty:
                            st.session_state.ss_final_signals_df_display = df_signals_full_history_combined.loc[common_index_display].copy()
                            st.success(f"Segnali finali filtrati per visualizzazione. Shape: {st.session_state.ss_final_signals_df_display.shape}")
                            logger.info(f"Segnali finali filtrati per visualizzazione. Shape: {st.session_state.ss_final_signals_df_display.shape}")
                        else:
                            st.warning("Nessun indice comune tra segnali calcolati e dati filtrati per visualizzazione.")
                            logger.warning("Nessun indice comune tra segnali e dati filtrati per visualizzazione.")
                            st.session_state.ss_final_signals_df_display = pd.DataFrame()
                    elif st.session_state.ss_raw_data_df_filtered is None or st.session_state.ss_raw_data_df_filtered.empty:
                        st.warning("Intervallo visualizzazione senza dati; segnali non mostrati per questo intervallo.")
                        logger.warning("Intervallo visualizzazione senza dati; segnali non mostrati per questo intervallo.")
                        st.session_state.ss_final_signals_df_display = pd.DataFrame() 
                    else: 
                        logger.info("Nessun filtro di visualizzazione applicato ai segnali (mostrando tutto lo storico dei segnali).")
                        st.session_state.ss_final_signals_df_display = df_signals_full_history_combined.copy()


                    if st.session_state.ss_final_signals_df_display is not None and not st.session_state.ss_final_signals_df_display.empty:
                        last_signal_row_display = st.session_state.ss_final_signals_df_display.iloc[-1]
                        st.session_state.ss_last_generated_signal_info = {
                            "ticker": st.session_state.ss_ticker_input,
                            "date": last_signal_row_display.name.strftime('%Y-%m-%d %H:%M:%S') if isinstance(last_signal_row_display.name, pd.Timestamp) else str(last_signal_row_display.name),
                            "ml_signal": last_signal_row_display.get('ml_signal', 'N/A'),
                            "breakout_signal": last_signal_row_display.get('breakout_signal', 'N/A'),
                            "close_price": f"{last_signal_row_display.get('Close', 0.0):.2f}" if 'Close' in last_signal_row_display else "N/A"
                        }
                        sound_config = CONFIG.get('sound_utils', {})
                        email_config = CONFIG.get('email_notifications', {})
                        if last_signal_row_display.get('ml_signal') == 'BUY':
                            play_buy_signal_sound(sound_config)
                            if email_config.get('enabled', False): send_signal_email_notification(st.session_state.ss_last_generated_signal_info, email_config, LOADED_SECRETS)
                        elif last_signal_row_display.get('ml_signal') == 'SELL':
                            play_sell_signal_sound(sound_config)
                            if email_config.get('enabled', False): send_signal_email_notification(st.session_state.ss_last_generated_signal_info, email_config, LOADED_SECRETS)
                else:
                    st.warning("Impossibile generare segnali: predizioni ML non disponibili o colonna predizioni mancante.")
                    logger.warning("Generazione segnali saltata: predizioni ML non disponibili.")
        else: 
             st.error("Elaborazione ML interrotta: nessun dato grezzo storico caricato o disponibile.")
             logger.error("Elaborazione ML interrotta: dati grezzi storici non disponibili.")
            
        progress_bar.progress(100, text="Analisi completata!")
        time.sleep(0.5) 
        progress_bar.empty() 

    # Resetta il flag ss_analysis_run_triggered ALLA FINE della pipeline condizionata
    if st.session_state.get('ss_analysis_run_triggered', False): 
        st.session_state.ss_analysis_run_triggered = False
        logger.debug("Flag ss_analysis_run_triggered resettato a False.")


# --- AREA PRINCIPALE PER VISUALIZZAZIONE RISULTATI ---
st.markdown("---")
st.header(f"📊 Risultati per: {st.session_state.ss_ticker_input if st.session_state.ss_ticker_input else 'N/D'}")

if st.session_state.ss_final_signals_df_display is not None and not st.session_state.ss_final_signals_df_display.empty:
    if st.session_state.ss_last_generated_signal_info:
        st.subheader("📢 Ultimo Segnale Generato (nell'intervallo visualizzato):")
        # ... (codice visualizzazione ultimo segnale come prima) ...
        sig_info = st.session_state.ss_last_generated_signal_info
        ml_color = "green" if sig_info['ml_signal'] == "BUY" else "red" if sig_info['ml_signal'] == "SELL" else "gray"
        breakout_color = "blue" if sig_info['breakout_signal'] == "BULLISH" else "orange" if sig_info['breakout_signal'] == "BEARISH" else "gray"
        st.markdown(f"""
        *   **Ticker:** `{sig_info['ticker']}`
        *   **Data Segnale:** `{sig_info['date']}`
        *   **Segnale ML:** <span style='color:{ml_color}; font-weight:bold;'>{sig_info['ml_signal']}</span>
        *   **Segnale Breakout:** <span style='color:{breakout_color};'>{sig_info['breakout_signal']}</span>
        *   **Prezzo Chiusura (al segnale):** `{sig_info['close_price']}`
        """, unsafe_allow_html=True)
        st.markdown("---")

    st.subheader("📈 Grafico Interattivo con Segnali")
    
    df_features_for_chart_display = pd.DataFrame() 
    # Per il grafico, usiamo ss_features_df_full_history MA filtrato per l'intervallo di visualizzazione (ss_raw_data_df_filtered.index)
    if st.session_state.ss_features_df_full_history is not None and not st.session_state.ss_features_df_full_history.empty and \
       st.session_state.ss_raw_data_df_filtered is not None and not st.session_state.ss_raw_data_df_filtered.empty:
        
        common_idx_chart = st.session_state.ss_raw_data_df_filtered.index.intersection(st.session_state.ss_features_df_full_history.index)
        if not common_idx_chart.empty:
            df_features_for_chart_display = st.session_state.ss_features_df_full_history.loc[common_idx_chart].copy()
            logger.debug(f"DataFrame per grafico (df_features_for_chart_display) creato. Shape: {df_features_for_chart_display.shape}")
        else:
            st.warning("Nessun indice comune tra dati filtrati e feature per il grafico. Impossibile creare il grafico principale.")
            logger.warning("Nessun indice comune per df_features_for_chart_display.")

    if not df_features_for_chart_display.empty:
        chart_fig = create_main_stock_chart(
            df_ohlcv_ma=df_features_for_chart_display, 
            df_signals=st.session_state.ss_final_signals_df_display, 
            ticker=st.session_state.ss_ticker_input,
            ma_periods_to_show=CONFIG.get('visualization',{}).get('show_ma_periods', [20, 50])
        )
        st.plotly_chart(chart_fig, use_container_width=True)
    else:
        # Non mostrare questo warning se l'analisi non è ancora stata eseguita o se i dati grezzi non sono stati caricati.
        # Mostra solo se l'analisi è stata tentata e df_features_for_chart_display è risultato vuoto.
        if st.session_state.get('ss_raw_data_df_full_history') is not None: # Indica che un tentativo di caricamento è stato fatto
             st.warning("Dati insufficienti o non allineati per visualizzare il grafico principale.")
             logger.warning("Dati insufficienti/non allineati per grafico principale.")


    with st.expander("👁️ Visualizza Dati Tabellari Dettagliati (ultimi 100 record dell'intervallo visualizzato)"):
        if st.session_state.ss_raw_data_df_filtered is not None and not st.session_state.ss_raw_data_df_filtered.empty: 
            st.markdown("#### Dati Grezzi (OHLCV - Intervallo Visualizzato)")
            st.dataframe(st.session_state.ss_raw_data_df_filtered.tail(100))
        
        # Mostra ss_features_df_full_history filtrato per l'intervallo di visualizzazione
        if st.session_state.ss_features_df_full_history is not None and \
           st.session_state.ss_raw_data_df_filtered is not None and not st.session_state.ss_raw_data_df_filtered.empty:
            common_idx_feat_tbl = st.session_state.ss_raw_data_df_filtered.index.intersection(st.session_state.ss_features_df_full_history.index)
            if not common_idx_feat_tbl.empty:
                 st.markdown("#### Feature Tecniche Calcolate (Intervallo Visualizzato)")
                 st.dataframe(st.session_state.ss_features_df_full_history.loc[common_idx_feat_tbl].tail(100))
        
        # Mostra ss_target_and_preds_df_full_history filtrato
        if st.session_state.ss_target_and_preds_df_full_history is not None and \
           st.session_state.ss_raw_data_df_filtered is not None and not st.session_state.ss_raw_data_df_filtered.empty:
            common_idx_pred_tbl = st.session_state.ss_raw_data_df_filtered.index.intersection(st.session_state.ss_target_and_preds_df_full_history.index)
            if not common_idx_pred_tbl.empty:
                st.markdown("#### Target di Predizione e Predizioni ML (Intervallo Visualizzato)")
                st.dataframe(st.session_state.ss_target_and_preds_df_full_history.loc[common_idx_pred_tbl].tail(100))
        
        if st.session_state.ss_final_signals_df_display is not None and not st.session_state.ss_final_signals_df_display.empty: 
            st.markdown("#### Segnali Finali (ML e Breakout - Intervallo Visualizzato)")
            st.dataframe(st.session_state.ss_final_signals_df_display.tail(100))


elif st.session_state.get('ss_ticker_input'): 
    if 'ss_analysis_run_triggered' not in st.session_state or not st.session_state.ss_analysis_run_triggered : # Solo se non è in corso un'analisi
        # Questo messaggio appare se c'è un ticker ma non ci sono risultati (es. dopo un errore nella pipeline precedente, o al primo caricamento)
        # logger.debug("Area risultati: Nessun segnale finale disponibile, ma ticker presente. Controllo dati grezzi...")
        if st.session_state.get('ss_raw_data_df_full_history') is None and st.session_state.get('ss_ticker_input'): 
             st.warning("Dati non ancora caricati o analisi fallita. Controlla i log di processo sopra se hai eseguito un'analisi.")
        elif st.session_state.get('ss_raw_data_df_filtered') is not None and st.session_state.get('ss_raw_data_df_filtered').empty : 
             st.warning("Nessun dato grezzo disponibile per il ticker e l'intervallo di visualizzazione selezionato.")
else: 
    st.info("👋 Benvenuto! Inserisci i parametri nella sidebar a sinistra e clicca 'Analizza e Genera Segnali' per iniziare.")

st.markdown("---")
st.caption(f"Dashboard v{APP_VERSION_FROM_CONFIG}. Controlla i log del server per dettagli DEBUG/INFO.")
st.caption(f"Ultimo aggiornamento dell'interfaccia Streamlit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}")

with st.sidebar.expander("🔍 DEBUG: Stato Sessione Completo", expanded=False):
    session_state_dict_for_json = {}
    for k, v_item in st.session_state.to_dict().items(): # Rinominato v a v_item
        if isinstance(v_item, pd.DataFrame):
            session_state_dict_for_json[k] = f"DataFrame with shape {v_item.shape}" if v_item is not None else "None"
        elif isinstance(v_item, (datetime, pd.Timestamp, pd.Period)):
             session_state_dict_for_json[k] = str(v_item)
        else:
            try: 
                json.dumps(v_item) 
                session_state_dict_for_json[k] = v_item
            except (TypeError, OverflowError): 
                session_state_dict_for_json[k] = str(v_item) 
    st.json(session_state_dict_for_json, expanded=False)
