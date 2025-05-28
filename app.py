# app.py - v1.6.5 (Main Application Orchestrator - Improved data loading logic)
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import yaml 
import os 
import time # Aggiunto per il progress_bar.empty()

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

try:
    with open(CONFIG_FILE, 'r') as f:
        CONFIG = yaml.safe_load(f)
    APP_VERSION_FROM_CONFIG = CONFIG.get('version', 'v1.6.5-spec-impl (config fallback)')
    config_loaded_successfully_flag = True
except FileNotFoundError:
    APP_VERSION_FROM_CONFIG = "ERRORE CONFIG" 
    pass 
except yaml.YAMLError as e:
    APP_VERSION_FROM_CONFIG = "ERRORE CONFIG YAML"
    yaml_error_message_for_later = e 
    pass

# --- CONFIGURAZIONE PAGINA STREAMLIT ---
st.set_page_config(
    layout="wide",
    page_title=f"Stock Signal Dashboard {APP_VERSION_FROM_CONFIG}", 
    page_icon="📈" 
)

if config_loaded_successfully_flag:
    print(f"INFO [app.py_module]: {CONFIG_FILE} caricato correttamente. Versione da config: {APP_VERSION_FROM_CONFIG}")
    if 'config_loaded_successfully' not in st.session_state: 
        st.session_state.config_loaded_successfully = True
elif 'yaml_error_message_for_later' in locals():
    st.error(f"ERRORE CRITICO nel parsing di '{CONFIG_FILE}': {yaml_error_message_for_later}. L'applicazione non può continuare.")
    st.session_state.config_loaded_successfully = False
    st.stop()
else: 
    st.error(f"ERRORE CRITICO: Il file di configurazione '{CONFIG_FILE}' non è stato trovato. L'applicazione non può continuare.")
    st.session_state.config_loaded_successfully = False
    st.stop()

# --- GESTIONE CHIAVI API (da st.secrets) ---
ALPHA_VANTAGE_API_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY")
if not ALPHA_VANTAGE_API_KEY and CONFIG.get('alpha_vantage'): 
    st.sidebar.warning("Chiave API Alpha Vantage (ALPHA_VANTAGE_API_KEY) non trovata in st.secrets.")

GOOGLE_AI_STUDIO_URL = None
GOOGLE_AI_STUDIO_TOKEN = None
if CONFIG.get('ml_model', {}).get('use_google_ai_studio', False):
    url_secret_name = CONFIG.get('ml_model', {}).get('google_ai_studio_url_secret_name')
    token_secret_name = CONFIG.get('ml_model', {}).get('google_ai_studio_token_secret_name')
    if url_secret_name and token_secret_name:
        GOOGLE_AI_STUDIO_URL = st.secrets.get(url_secret_name)
        GOOGLE_AI_STUDIO_TOKEN = st.secrets.get(token_secret_name)
        if not GOOGLE_AI_STUDIO_URL or not GOOGLE_AI_STUDIO_TOKEN:
            st.sidebar.warning(f"Google AI Studio: URL ({url_secret_name}) o Token ({token_secret_name}) mancanti in st.secrets.")
    else:
        st.sidebar.error("Nomi secret Google AI Studio non in config.yaml.")

EMAIL_SMTP_PASSWORD = None
if CONFIG.get('email_notifications', {}).get('enabled', False):
    pwd_secret_name = CONFIG.get('email_notifications', {}).get('smtp_password_secret_name')
    if pwd_secret_name:
        EMAIL_SMTP_PASSWORD = st.secrets.get(pwd_secret_name)
        if not EMAIL_SMTP_PASSWORD:
            st.sidebar.warning(f"Email: password SMTP ({pwd_secret_name}) non in st.secrets.")
    else:
        st.sidebar.error("Nome secret password SMTP non in config.yaml.")

LOADED_SECRETS = {
    "ALPHA_VANTAGE_API_KEY": ALPHA_VANTAGE_API_KEY,
    CONFIG.get('ml_model', {}).get('google_ai_studio_url_secret_name', 'GOOGLE_AI_STUDIO_URL_PLACEHOLDER'): GOOGLE_AI_STUDIO_URL,
    CONFIG.get('ml_model', {}).get('google_ai_studio_token_secret_name', 'GOOGLE_AI_STUDIO_TOKEN_PLACEHOLDER'): GOOGLE_AI_STUDIO_TOKEN,
    CONFIG.get('email_notifications', {}).get('smtp_password_secret_name', 'EMAIL_SMTP_PASSWORD_PLACEHOLDER'): EMAIL_SMTP_PASSWORD
}

# --- STATO DELLA SESSIONE (SESSION STATE) ---
default_session_state_values = {
    'ss_ticker_input': "AAPL", 
    'ss_asset_type': "stock", 
    # Imposta una start_date di default che permetta di avere abbastanza dati per indicatori e training
    # Es: MA50 + 90 giorni training + buffer = ~150-200 giorni prima della start_date effettiva dell'analisi
    'ss_start_date_display': datetime.now().date() - timedelta(days=90), # Per l'intervallo di visualizzazione utente
    'ss_end_date_display': datetime.now().date(),     # Per l'intervallo di visualizzazione utente
    'ss_raw_data_df_full_history': None, # Conterrà lo storico più lungo caricato
    'ss_raw_data_df_filtered': None, # Filtrato per l'intervallo utente
    'ss_features_df': None,         
    'ss_target_and_preds_df': None, 
    'ss_final_signals_df': None,   
    'ss_trained_ml_model': None,     
    'ss_last_generated_signal_info': None, 
    'ss_analysis_run_triggered': False 
}
for key, value in default_session_state_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- TITOLO E HEADER DELL'APPLICAZIONE ---
st.title(f"📈 Stock & Crypto Signal Dashboard")
st.markdown(f"**Versione:** `{APP_VERSION_FROM_CONFIG}` - _Basato sulle specifiche del progetto._")
st.markdown("---")

# --- SIDEBAR PER INPUT UTENTE E CONTROLLI ---
with st.sidebar:
    st.header("🛠️ Controlli e Parametri")

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
        # Rinominato ss_start_date a ss_start_date_display
        st.session_state.ss_start_date_display = st.date_input("Data Inizio", value=st.session_state.ss_start_date_display, key="start_date_input_display", help="Data inizio per visualizzazione e analisi finale.")
    with col_date2:
        # Rinominato ss_end_date a ss_end_date_display
        st.session_state.ss_end_date_display = st.date_input("Data Fine", value=st.session_state.ss_end_date_display, key="end_date_input_display", help="Data fine. Deve essere >= Data Inizio.")

    if st.session_state.ss_start_date_display > st.session_state.ss_end_date_display:
        st.error("Errore: La Data di Inizio deve essere precedente o uguale alla Data di Fine.")
        st.stop() 

    st.markdown("---") 
    if st.button("🚀 Analizza e Genera Segnali", type="primary", use_container_width=True, key="run_analysis_button"):
        if not st.session_state.ss_ticker_input:
            st.warning("Per favore, inserisci un simbolo Ticker o Coin ID valido.")
        else:
            st.session_state.ss_analysis_run_triggered = True 
            st.session_state.ss_raw_data_df_full_history = None # Rinominato
            st.session_state.ss_raw_data_df_filtered = None   # Nuovo
            st.session_state.ss_features_df = None
            st.session_state.ss_target_and_preds_df = None
            st.session_state.ss_final_signals_df = None
            st.session_state.ss_trained_ml_model = None
            st.session_state.ss_last_generated_signal_info = None
            st.info(f"Avvio analisi per {st.session_state.ss_ticker_input} ({st.session_state.ss_asset_type})...")
    st.markdown("---")

# --- PIPELINE DI ELABORAZIONE DATI E SEGNALI ---
if st.session_state.ss_analysis_run_triggered:
    log_container = st.container()
    log_container.markdown("### ⚙️ Log di Processo dell'Analisi")
    
    with log_container:
        progress_bar = st.progress(0, text="Inizio analisi...")

        # Definisci un numero minimo di giorni di storico da caricare per indicatori e training
        # Es. MA più lunga (50) + giorni di training (90) + buffer (es. 60 per dropna, shift target, ecc.) = ~200 giorni
        MIN_DAYS_HISTORY_FOR_ML = CONFIG.get('ml_model', {}).get('min_days_for_indicators_and_training', 200)
        
        # Calcola la data di inizio effettiva per il caricamento dati
        # basata sulla data di fine scelta dall'utente e MIN_DAYS_HISTORY_FOR_ML
        actual_data_load_start_date = st.session_state.ss_end_date_display - timedelta(days=MIN_DAYS_HISTORY_FOR_ML -1) # -1 perché "days" include start e end

        # 1. CARICAMENTO DATI GREZZI (OHLCV) - Carica uno storico più lungo
        progress_bar.progress(10, text=f"Caricamento storico esteso per {st.session_state.ss_ticker_input}...")
        if st.session_state.ss_asset_type == "stock":
            if not ALPHA_VANTAGE_API_KEY:
                st.error("Impossibile caricare dati azioni: Chiave API Alpha Vantage mancante.")
                st.session_state.ss_raw_data_df_full_history = None 
            else:
                # Per AlphaVantage, 'full' dovrebbe dare abbastanza storico.
                # Non modifichiamo le date di richiesta qui, ci affidiamo a 'full'.
                st.session_state.ss_raw_data_df_full_history = get_stock_data(
                    ALPHA_VANTAGE_API_KEY, 
                    st.session_state.ss_ticker_input,
                    CONFIG.get('alpha_vantage', {}).get('function', 'TIME_SERIES_DAILY_ADJUSTED'),
                    "full" # Forza 'full' per avere il massimo storico possibile
                )
        elif st.session_state.ss_asset_type == "crypto":
            # Per CoinGecko, calcoliamo i giorni da oggi fino a actual_data_load_start_date
            # Questo assicura che carichiamo abbastanza dati se l'endpoint lo permette
            days_to_fetch_for_coingecko = (datetime.now().date() - actual_data_load_start_date).days + 1
            if days_to_fetch_for_coingecko <=0: days_to_fetch_for_coingecko = MIN_DAYS_HISTORY_FOR_ML # Fallback
            
            st.write(f"DEBUG [app.py]: Caricamento crypto - Data inizio richiesta API: {actual_data_load_start_date}, Giorni da fetchare: {days_to_fetch_for_coingecko}")

            st.session_state.ss_raw_data_df_full_history = get_crypto_data(
                st.session_state.ss_ticker_input,
                vs_currency=CONFIG.get('coingecko',{}).get('vs_currency', 'usd'),
                days=days_to_fetch_for_coingecko # Usa i giorni calcolati per avere uno storico sufficiente
            )
        
        # Ora FILTRA lo storico completo per l'INTERVALLO DI VISUALIZZAZIONE scelto dall'utente
        if st.session_state.ss_raw_data_df_full_history is not None and not st.session_state.ss_raw_data_df_full_history.empty:
            start_dt_display_filter = pd.to_datetime(st.session_state.ss_start_date_display)
            end_dt_display_filter = pd.to_datetime(st.session_state.ss_end_date_display)
            
            if not isinstance(st.session_state.ss_raw_data_df_full_history.index, pd.DatetimeIndex):
                st.session_state.ss_raw_data_df_full_history.index = pd.to_datetime(st.session_state.ss_raw_data_df_full_history.index)

            # Il DataFrame per l'analisi e la visualizzazione sarà ss_raw_data_df_filtered
            st.session_state.ss_raw_data_df_filtered = st.session_state.ss_raw_data_df_full_history[
                (st.session_state.ss_raw_data_df_full_history.index >= start_dt_display_filter) & 
                (st.session_state.ss_raw_data_df_full_history.index <= end_dt_display_filter)
            ].copy() # Usa .copy() per evitare SettingWithCopyWarning più avanti

            if st.session_state.ss_raw_data_df_filtered.empty:
                st.warning(f"Nessun dato per {st.session_state.ss_ticker_input} trovato nell'intervallo di visualizzazione ({st.session_state.ss_start_date_display.strftime('%Y-%m-%d')} - {st.session_state.ss_end_date_display.strftime('%Y-%m-%d')}) dopo aver caricato lo storico esteso.")
                # Non invalidare ss_raw_data_df_full_history, potrebbe servire per il training se l'intervallo utente è troppo breve
            else:
                st.success(f"Storico esteso caricato. Dati per visualizzazione/analisi ({st.session_state.ss_start_date_display.strftime('%Y-%m-%d')} - {st.session_state.ss_end_date_display.strftime('%Y-%m-%d')}): Shape {st.session_state.ss_raw_data_df_filtered.shape}")
        
        elif st.session_state.ss_raw_data_df_full_history is None: 
            st.error(f"Fallimento nel caricamento dello storico esteso per {st.session_state.ss_ticker_input}.")
        
        # --- INIZIO ELABORAZIONE ML ---
        # Usiamo ss_raw_data_df_full_history per calcolare feature e allenare il modello,
        # per avere abbastanza dati per le finestre degli indicatori e il training.
        # Le predizioni e i segnali verranno poi filtrati sull'intervallo di visualizzazione.
        
        # Verifica se ss_raw_data_df_full_history è valido prima di procedere con ML
        if st.session_state.ss_raw_data_df_full_history is not None and not st.session_state.ss_raw_data_df_full_history.empty:
            # Verifica se lo storico caricato è sufficiente
            if len(st.session_state.ss_raw_data_df_full_history) < MIN_DAYS_HISTORY_FOR_ML / 2: # Soglia arbitraria, es. metà del minimo richiesto
                st.warning(f"Storico caricato ({len(st.session_state.ss_raw_data_df_full_history)} righe) potrebbe essere insufficiente per un'analisi ML robusta (minimo richiesto: {MIN_DAYS_HISTORY_FOR_ML}). I risultati potrebbero essere inaffidabili.")
            
            progress_bar.progress(25, text="Calcolo feature tecniche su storico esteso...")
            # 2. CALCOLO FEATURE TECNICHE (sullo storico completo)
            st.session_state.ss_features_df = calculate_technical_features(st.session_state.ss_raw_data_df_full_history)
            
            if st.session_state.ss_features_df.empty or len(st.session_state.ss_features_df) < 10: # Aggiunto controllo lunghezza
                st.error("Fallimento nel calcolo delle feature tecniche o dati insufficienti dopo il calcolo.")
            else:
                st.success(f"Feature tecniche calcolate su storico esteso. Shape: {st.session_state.ss_features_df.shape}")

                progress_bar.progress(40, text="Creazione target di predizione...")
                # 3. CREAZIONE TARGET PER PREDIZIONE ML
                pred_horizon = CONFIG.get('ml_model', {}).get('prediction_target_horizon_days', 3)
                df_with_target_full_hist = create_prediction_targets(st.session_state.ss_features_df, horizon=pred_horizon) # Usa ss_features_df (che è basato su full_history)
                target_col_name = f'target_{pred_horizon}d_pct_change'

                # 4. TRAINING MODELLO ML E PREDIZIONI
                feature_cols_for_ml = CONFIG.get('ml_model',{}).get('feature_columns_for_training', ['MA20', 'MA50', 'RSI', 'StochRSI_K', 'Momentum_ROC10', 'ADX', 'MACD_line'])
                feature_cols_for_ml = [col for col in feature_cols_for_ml if col in df_with_target_full_hist.columns]

                if not feature_cols_for_ml:
                    st.error("Nessuna colonna feature valida trovata per il training/predizione ML.")
                elif target_col_name not in df_with_target_full_hist.columns:
                    st.error(f"Colonna target '{target_col_name}' non creata o mancante per il training ML.")
                else:
                    predictions_series = None 
                    if CONFIG.get('ml_model', {}).get('use_google_ai_studio', False):
                        # ... (logica AI Studio) ...
                        st.warning("Integrazione Google AI Studio non completamente implementata.")
                        predictions_series = pd.Series(index=df_with_target_full_hist.index, dtype=float) 
                    else: 
                        progress_bar.progress(55, text="Training modello RandomForest...")
                        st.session_state.ss_trained_ml_model = train_random_forest_model(
                            df_with_target_full_hist, 
                            feature_columns=feature_cols_for_ml,
                            target_column=target_col_name,
                            n_estimators=CONFIG.get('ml_model',{}).get('random_forest_n_estimators', 100)
                        )
                        if st.session_state.ss_trained_ml_model:
                            st.success("Modello RandomForest addestrato.")
                            progress_bar.progress(70, text="Generazione predizioni ML...")
                            predictions_series = generate_model_predictions(
                                st.session_state.ss_trained_ml_model,
                                df_with_target_full_hist, # Predici su tutto lo storico con feature
                                feature_columns=feature_cols_for_ml
                            )
                        else:
                            st.error("Fallimento training modello RandomForest.")
                            
                    if predictions_series is not None:
                        st.session_state.ss_target_and_preds_df = df_with_target_full_hist.copy()
                        prediction_col_ml_name = f'prediction_{pred_horizon}d_pct_change'
                        st.session_state.ss_target_and_preds_df[prediction_col_ml_name] = predictions_series
                        st.success(f"Predizioni ML generate e aggiunte come '{prediction_col_ml_name}' su storico esteso.")
                    else:
                        st.error("Fallimento nella generazione delle predizioni ML.")

                # 5. GENERAZIONE SEGNALI DI TRADING (su ss_target_and_preds_df che è basato su full_history)
                if st.session_state.ss_target_and_preds_df is not None and prediction_col_ml_name in st.session_state.ss_target_and_preds_df:
                    progress_bar.progress(85, text="Generazione segnali di trading...")
                    df_ml_signals_only = generate_signals_from_ml_predictions(
                        st.session_state.ss_target_and_preds_df,
                        prediction_column_name=prediction_col_ml_name,
                        buy_threshold=CONFIG.get('signal_logic',{}).get('buy_threshold_change', 0.005),
                        sell_threshold=CONFIG.get('signal_logic',{}).get('sell_threshold_change', -0.005)
                    )
                    # Per i breakout, usiamo ss_features_df (basato su full_history)
                    df_breakout_signals_only = detect_breakout_signals(
                        st.session_state.ss_features_df, 
                        high_low_period=CONFIG.get('signal_logic',{}).get('breakout_days_high_low_period', 20),
                        volume_avg_factor=CONFIG.get('signal_logic',{}).get('breakout_volume_avg_factor', 1.0),
                        volume_period=CONFIG.get('signal_logic',{}).get('breakout_days_high_low_period', 20) 
                    )
                    # Combina segnali (entrambi basati su full_history)
                    df_signals_full_history = combine_signals(df_ml_signals_only, df_breakout_signals_only)
                    df_signals_full_history = apply_trading_spreads(
                        df_signals_full_history,
                        st.session_state.ss_asset_type,
                        CONFIG.get('spreads', {})
                    )
                    
                    # Ora FILTRA i segnali finali per l'intervallo di visualizzazione dell'utente
                    if not df_signals_full_history.empty and st.session_state.ss_raw_data_df_filtered is not None:
                        st.session_state.ss_final_signals_df = df_signals_full_history.loc[
                            st.session_state.ss_raw_data_df_filtered.index
                        ].copy() # Usa .copy()
                        st.success(f"Segnali di trading finali generati e filtrati per visualizzazione. Shape: {st.session_state.ss_final_signals_df.shape}")
                    elif st.session_state.ss_raw_data_df_filtered is None or st.session_state.ss_raw_data_df_filtered.empty:
                        st.warning("Intervallo di visualizzazione non ha dati, quindi non ci sono segnali da mostrare per questo intervallo, anche se potrebbero esistere nello storico.")
                        st.session_state.ss_final_signals_df = pd.DataFrame() # Vuoto se non c'è intervallo di visualizzazione
                    else:
                        st.session_state.ss_final_signals_df = df_signals_full_history # Mostra tutto se il filtro fallisce

                    if st.session_state.ss_final_signals_df is not None and not st.session_state.ss_final_signals_df.empty:
                        last_signal_row = st.session_state.ss_final_signals_df.iloc[-1]
                        st.session_state.ss_last_generated_signal_info = {
                            "ticker": st.session_state.ss_ticker_input,
                            "date": last_signal_row.name.strftime('%Y-%m-%d %H:%M:%S') if isinstance(last_signal_row.name, pd.Timestamp) else str(last_signal_row.name),
                            "ml_signal": last_signal_row.get('ml_signal', 'N/A'),
                            "breakout_signal": last_signal_row.get('breakout_signal', 'N/A'),
                            "close_price": f"{last_signal_row.get('Close', 0.0):.2f}" if 'Close' in last_signal_row else "N/A"
                        }
                        # ... (suoni e notifiche email come prima) ...
                        sound_config = CONFIG.get('sound_utils', {})
                        email_config = CONFIG.get('email_notifications', {})
                        if last_signal_row.get('ml_signal') == 'BUY':
                            play_buy_signal_sound(sound_config)
                            if email_config.get('enabled', False): send_signal_email_notification(st.session_state.ss_last_generated_signal_info, email_config, LOADED_SECRETS)
                        elif last_signal_row.get('ml_signal') == 'SELL':
                            play_sell_signal_sound(sound_config)
                            if email_config.get('enabled', False): send_signal_email_notification(st.session_state.ss_last_generated_signal_info, email_config, LOADED_SECRETS)

                else:
                    st.warning("Impossibile generare segnali: predizioni ML non disponibili o colonna predizioni mancante.")
        else: # Se ss_raw_data_df_full_history è None o vuoto
             st.error("Elaborazione ML interrotta: nessun dato grezzo storico caricato o disponibile.")
            
        progress_bar.progress(100, text="Analisi completata!")
        time.sleep(1) 
        progress_bar.empty() 

    if st.session_state.ss_analysis_run_triggered: 
        st.session_state.ss_analysis_run_triggered = False

# --- AREA PRINCIPALE PER VISUALIZZAZIONE RISULTATI ---
# Ora la visualizzazione usa ss_raw_data_df_filtered e ss_final_signals_df (che è già filtrato)
st.markdown("---")
st.header(f"📊 Risultati per: {st.session_state.ss_ticker_input if st.session_state.ss_ticker_input else 'N/D'}")

if st.session_state.ss_final_signals_df is not None and not st.session_state.ss_final_signals_df.empty:
    if st.session_state.ss_last_generated_signal_info:
        # ... (codice visualizzazione ultimo segnale come prima) ...
        st.subheader("📢 Ultimo Segnale Generato (nell'intervallo visualizzato):")
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
    # Per il grafico, usiamo ss_features_df ma filtrato sull'intervallo di visualizzazione
    # e ss_final_signals_df (già filtrato)
    if st.session_state.ss_features_df is not None and not st.session_state.ss_features_df.empty and \
       st.session_state.ss_raw_data_df_filtered is not None and not st.session_state.ss_raw_data_df_filtered.empty:
        
        # Filtra ss_features_df per l'intervallo di visualizzazione per il grafico
        df_features_for_chart = st.session_state.ss_features_df.loc[
            st.session_state.ss_raw_data_df_filtered.index
        ].copy()

        chart_fig = create_main_stock_chart(
            df_ohlcv_ma=df_features_for_chart, # Feature calcolate su storico, poi filtrate per display
            df_signals=st.session_state.ss_final_signals_df, # Segnali già filtrati per display
            ticker=st.session_state.ss_ticker_input,
            ma_periods_to_show=CONFIG.get('visualization',{}).get('show_ma_periods', [20, 50])
        )
        st.plotly_chart(chart_fig, use_container_width=True)
    else:
        st.warning("Dati per il grafico (feature DF filtrato o segnali finali) non disponibili.")

    with st.expander("👁️ Visualizza Dati Tabellari Dettagliati (ultimi 100 record dell'intervallo visualizzato)"):
        if st.session_state.ss_raw_data_df_filtered is not None: # Mostra il filtrato
            st.markdown("#### Dati Grezzi (OHLCV - Intervallo Visualizzato)")
            st.dataframe(st.session_state.ss_raw_data_df_filtered.tail(100))
        
        # Mostra le feature e le predizioni filtrate sull'intervallo di visualizzazione
        if st.session_state.ss_features_df is not None and st.session_state.ss_raw_data_df_filtered is not None:
            st.markdown("#### Feature Tecniche Calcolate (Intervallo Visualizzato)")
            st.dataframe(st.session_state.ss_features_df.loc[st.session_state.ss_raw_data_df_filtered.index].tail(100))
        
        if st.session_state.ss_target_and_preds_df is not None and st.session_state.ss_raw_data_df_filtered is not None:
            st.markdown("#### Target di Predizione e Predizioni ML (Intervallo Visualizzato)")
            st.dataframe(st.session_state.ss_target_and_preds_df.loc[st.session_state.ss_raw_data_df_filtered.index].tail(100))
        
        if st.session_state.ss_final_signals_df is not None: # Già filtrato
            st.markdown("#### Segnali Finali (ML e Breakout - Intervallo Visualizzato)")
            st.dataframe(st.session_state.ss_final_signals_df.tail(100))

elif st.session_state.get('ss_ticker_input'): 
    if 'ss_analysis_run_triggered' not in st.session_state or not st.session_state.ss_analysis_run_triggered :
        if st.session_state.get('ss_raw_data_df_full_history') is None and st.session_state.get('ss_ticker_input'): # Modificato per controllare full_history
             st.warning("Dati non ancora caricati o analisi fallita. Controlla i log di processo sopra se hai eseguito un'analisi.")
        elif st.session_state.get('ss_raw_data_df_filtered') is not None and st.session_state.get('ss_raw_data_df_filtered').empty : # Controlla il filtrato
             st.warning("Nessun dato grezzo disponibile per il ticker e l'intervallo di visualizzazione selezionato.")
else: 
    st.info("👋 Benvenuto! Inserisci i parametri nella sidebar a sinistra e clicca 'Analizza e Genera Segnali' per iniziare.")

# ... (Footer e Debug Session State come prima) ...
st.markdown("---")
st.caption(f"Dashboard v{APP_VERSION_FROM_CONFIG}. Controlla i messaggi DEBUG e gli errori nel log di processo per dettagli sull'esecuzione.")
st.caption(f"Ultimo aggiornamento dell'interfaccia Streamlit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}")

with st.sidebar.expander("🔍 DEBUG: Stato Sessione Completo", expanded=False):
    session_state_dict_for_json = {}
    for k, v in st.session_state.to_dict().items():
        if isinstance(v, pd.DataFrame):
            session_state_dict_for_json[k] = f"DataFrame with shape {v.shape}" if v is not None else "None"
        elif isinstance(v, (datetime, pd.Timestamp, pd.Period)): # Aggiunto pd.Period
             session_state_dict_for_json[k] = str(v) # Usa str() per più tipi di data/ora
        else:
            try: # Prova a serializzare, altrimenti usa str()
                pd.io.json.dumps(v) # Test di serializzabilità
                session_state_dict_for_json[k] = v
            except TypeError:
                session_state_dict_for_json[k] = str(v)
    st.json(session_state_dict_for_json, expanded=False)
