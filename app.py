# app.py - v1.6.5 (Main Application Orchestrator)
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import yaml # Per caricare config.yaml
import os # Per controllare l'esistenza di file (es. suoni)

# Importa dai moduli del progetto
from data_utils import get_stock_data, get_crypto_data
from ml_model import calculate_technical_features, create_prediction_targets, train_random_forest_model, generate_model_predictions, get_predictions_from_ai_studio
from signal_logic import generate_signals_from_ml_predictions, detect_breakout_signals, apply_trading_spreads, combine_signals, send_signal_email_notification
from visualization import create_main_stock_chart
from sound_utils import play_buy_signal_sound, play_sell_signal_sound

# --- CARICAMENTO CONFIGURAZIONE ---
# Questa parte NON deve usare comandi st.* PRIMA di st.set_page_config()
CONFIG_FILE = "config.yaml"
CONFIG = {}
APP_VERSION_FROM_CONFIG = "N/A" # Valore di fallback
config_loaded_successfully_flag = False # Flag per tracciare il caricamento

try:
    with open(CONFIG_FILE, 'r') as f:
        CONFIG = yaml.safe_load(f)
    APP_VERSION_FROM_CONFIG = CONFIG.get('version', 'v1.6.5-spec-impl (config fallback)')
    config_loaded_successfully_flag = True
    # Il messaggio di debug verrà spostato DOPO st.set_page_config
except FileNotFoundError:
    # Gestiremo l'errore DOPO st.set_page_config per evitare crash qui
    APP_VERSION_FROM_CONFIG = "ERRORE CONFIG" 
    # Non possiamo usare st.error() qui
    pass 
except yaml.YAMLError as e:
    APP_VERSION_FROM_CONFIG = "ERRORE CONFIG YAML"
    # Non possiamo usare st.error() qui
    # Salva l'errore per mostrarlo dopo
    yaml_error_message_for_later = e 
    pass

# --- CONFIGURAZIONE PAGINA STREAMLIT ---
# DEVE ESSERE IL PRIMO COMANDO STREAMLIT!
st.set_page_config(
    layout="wide",
    page_title=f"Stock Signal Dashboard {APP_VERSION_FROM_CONFIG}", # Usa la versione caricata
    page_icon="📈" # Emoji per icona
)

# Ora possiamo usare comandi st.* per mostrare errori di caricamento config o messaggi di debug
if config_loaded_successfully_flag:
    st.write(f"DEBUG [app.py]: {CONFIG_FILE} caricato correttamente. Versione da config: {APP_VERSION_FROM_CONFIG}")
    # Inizializza st.session_state.config_loaded_successfully qui, DOPO st.set_page_config
    if 'config_loaded_successfully' not in st.session_state: # Solo se non già impostato
        st.session_state.config_loaded_successfully = True
elif 'yaml_error_message_for_later' in locals():
    st.error(f"ERRORE CRITICO nel parsing di '{CONFIG_FILE}': {yaml_error_message_for_later}. L'applicazione non può continuare.")
    st.session_state.config_loaded_successfully = False
    st.stop()
else: # FileNotFoundError
    st.error(f"ERRORE CRITICO: Il file di configurazione '{CONFIG_FILE}' non è stato trovato. L'applicazione non può continuare.")
    st.session_state.config_loaded_successfully = False
    st.stop()


# --- GESTIONE CHIAVI API (da st.secrets) ---
# Alpha Vantage
ALPHA_VANTAGE_API_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY")
if not ALPHA_VANTAGE_API_KEY and CONFIG.get('alpha_vantage'): 
    st.sidebar.warning("Chiave API Alpha Vantage (ALPHA_VANTAGE_API_KEY) non trovata in st.secrets. Le funzionalità per le azioni potrebbero non funzionare.")

# Google AI Studio (opzionale)
GOOGLE_AI_STUDIO_URL = None
GOOGLE_AI_STUDIO_TOKEN = None
if CONFIG.get('ml_model', {}).get('use_google_ai_studio', False):
    url_secret_name = CONFIG.get('ml_model', {}).get('google_ai_studio_url_secret_name')
    token_secret_name = CONFIG.get('ml_model', {}).get('google_ai_studio_token_secret_name')
    if url_secret_name and token_secret_name:
        GOOGLE_AI_STUDIO_URL = st.secrets.get(url_secret_name)
        GOOGLE_AI_STUDIO_TOKEN = st.secrets.get(token_secret_name)
        if not GOOGLE_AI_STUDIO_URL or not GOOGLE_AI_STUDIO_TOKEN:
            st.sidebar.warning(f"Google AI Studio abilitato in config, ma URL ({url_secret_name}) o Token ({token_secret_name}) mancanti in st.secrets.")
    else:
        st.sidebar.error("Nomi dei secret per Google AI Studio non specificati in config.yaml.")

# Email SMTP Password (opzionale)
EMAIL_SMTP_PASSWORD = None
if CONFIG.get('email_notifications', {}).get('enabled', False):
    pwd_secret_name = CONFIG.get('email_notifications', {}).get('smtp_password_secret_name')
    if pwd_secret_name:
        EMAIL_SMTP_PASSWORD = st.secrets.get(pwd_secret_name)
        if not EMAIL_SMTP_PASSWORD:
            st.sidebar.warning(f"Notifiche email abilitate, ma password SMTP ({pwd_secret_name}) non trovata in st.secrets.")
    else:
        st.sidebar.error("Nome del secret per la password SMTP non specificato in config.yaml.")

# Raccogli tutti i secrets caricati in un dizionario per passarlo alle funzioni se necessario
LOADED_SECRETS = {
    "ALPHA_VANTAGE_API_KEY": ALPHA_VANTAGE_API_KEY,
    # Usa i nomi dei secret da config per le chiavi del dizionario, se disponibili
    CONFIG.get('ml_model', {}).get('google_ai_studio_url_secret_name', 'GOOGLE_AI_STUDIO_URL_PLACEHOLDER'): GOOGLE_AI_STUDIO_URL,
    CONFIG.get('ml_model', {}).get('google_ai_studio_token_secret_name', 'GOOGLE_AI_STUDIO_TOKEN_PLACEHOLDER'): GOOGLE_AI_STUDIO_TOKEN,
    CONFIG.get('email_notifications', {}).get('smtp_password_secret_name', 'EMAIL_SMTP_PASSWORD_PLACEHOLDER'): EMAIL_SMTP_PASSWORD
}


# --- STATO DELLA SESSIONE (SESSION STATE) ---
default_session_state_values = {
    'ss_ticker_input': "AAPL", 
    'ss_asset_type': "stock", 
    'ss_start_date': datetime.now() - timedelta(days=CONFIG.get('ml_model', {}).get('training_days', 90) + 180),
    'ss_end_date': datetime.now(),
    'ss_raw_data_df': None,          
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
    # Aggiorna il valore di default per il ticker se il tipo di asset cambia e l'input è vuoto o è quello di default dell'altro tipo
    if 'prev_asset_type' not in st.session_state: st.session_state.prev_asset_type = st.session_state.ss_asset_type
    
    if st.session_state.prev_asset_type != st.session_state.ss_asset_type:
        st.session_state.ss_ticker_input = "AAPL" if st.session_state.ss_asset_type == "stock" else "bitcoin"
    st.session_state.prev_asset_type = st.session_state.ss_asset_type
        
    current_ticker_value = st.session_state.get('ss_ticker_input', "AAPL" if st.session_state.ss_asset_type == "stock" else "bitcoin")
    
    st.session_state.ss_ticker_input = st.text_input(
        f"Inserisci Simbolo Ticker ({'Azioni' if st.session_state.ss_asset_type == 'stock' else 'Crypto'}):",
        value=current_ticker_value,
        placeholder=default_ticker_placeholder,
        key="ticker_text_input"
    )
    if st.session_state.ss_asset_type == "stock":
        st.session_state.ss_ticker_input = st.session_state.ss_ticker_input.upper()
    else:
        st.session_state.ss_ticker_input = st.session_state.ss_ticker_input.lower()


    st.markdown("##### Intervallo Date per Analisi:")
    col_date1, col_date2 = st.columns(2)
    with col_date1:
        st.session_state.ss_start_date = st.date_input("Data Inizio", value=st.session_state.ss_start_date, key="start_date_input", help="Seleziona la data di inizio per l'analisi dei dati storici.")
    with col_date2:
        st.session_state.ss_end_date = st.date_input("Data Fine", value=st.session_state.ss_end_date, key="end_date_input", help="Seleziona la data di fine. Deve essere uguale o successiva alla data di inizio.")

    if st.session_state.ss_start_date > st.session_state.ss_end_date:
        st.error("Errore: La Data di Inizio deve essere precedente o uguale alla Data di Fine.")
        st.stop() 

    st.markdown("---") 
    if st.button("🚀 Analizza e Genera Segnali", type="primary", use_container_width=True, key="run_analysis_button"):
        if not st.session_state.ss_ticker_input:
            st.warning("Per favore, inserisci un simbolo Ticker o Coin ID valido.")
        else:
            st.session_state.ss_analysis_run_triggered = True 
            st.session_state.ss_raw_data_df = None
            st.session_state.ss_features_df = None
            st.session_state.ss_target_and_preds_df = None
            st.session_state.ss_final_signals_df = None
            st.session_state.ss_trained_ml_model = None
            st.session_state.ss_last_generated_signal_info = None
            st.info(f"Avvio analisi per {st.session_state.ss_ticker_input} ({st.session_state.ss_asset_type})...")
    st.markdown("---")


# --- PIPELINE DI ELABORAZIONE DATI E SEGNALI ---
if st.session_state.ss_analysis_run_triggered:
    # Contenitore per i log di processo
    log_container = st.container()
    log_container.markdown("### ⚙️ Log di Processo dell'Analisi")
    
    # Usa with per assicurare che il log_container venga aggiornato
    with log_container:
        progress_bar = st.progress(0, text="Inizio analisi...")

        # 1. CARICAMENTO DATI GREZZI (OHLCV)
        progress_bar.progress(10, text=f"Caricamento dati per {st.session_state.ss_ticker_input}...")
        if st.session_state.ss_asset_type == "stock":
            if not ALPHA_VANTAGE_API_KEY:
                st.error("Impossibile caricare dati azioni: Chiave API Alpha Vantage mancante.")
                st.session_state.ss_raw_data_df = None # Assicura che sia None se fallisce
            else:
                st.session_state.ss_raw_data_df = get_stock_data(
                    ALPHA_VANTAGE_API_KEY, 
                    st.session_state.ss_ticker_input,
                    CONFIG.get('alpha_vantage', {}).get('function', 'TIME_SERIES_DAILY_ADJUSTED'),
                    CONFIG.get('alpha_vantage', {}).get('outputsize', 'full') 
                )
        elif st.session_state.ss_asset_type == "crypto":
            # Calcola il numero di giorni di storico necessari
            # Deve essere sufficiente per l'intervallo selezionato E per i giorni di training del modello
            user_selected_days = (st.session_state.ss_end_date - st.session_state.ss_start_date).days
            training_days_needed = CONFIG.get('ml_model', {}).get('training_days', 90)
            buffer_days = 30 # Un po' di buffer per finestre mobili, ecc.
            
            # CoinGecko carica 'days' di storico indietro dalla data corrente.
            # Quindi, per avere dati fino a start_date, dobbiamo calcolare i giorni da oggi a start_date.
            days_from_today_to_start = (datetime.now().date() - st.session_state.ss_start_date).days
            
            # Vogliamo il massimo tra i giorni necessari per il training e i giorni fino a start_date
            # per essere sicuri di avere dati per l'intero periodo di interesse.
            # CoinGecko ha un limite sui giorni per l'endpoint ohlc (es. 90 giorni se non > 90).
            # Per dati più vecchi, /market_chart è più flessibile ma dà solo close/volume/market_cap.
            # La funzione get_crypto_data usa l'endpoint ohlc, che potrebbe essere limitato.
            # Per ora, usiamo un calcolo semplice, ma potrebbe necessitare aggiustamenti
            # in base ai limiti effettivi di CoinGecko e alla strategia di caricamento dati.
            days_to_fetch_cg = max(user_selected_days + training_days_needed + buffer_days, days_from_today_to_start + buffer_days, CONFIG.get('coingecko',{}).get('days_history', 90) )
            # CoinGecko OHLC endpoint ha un limite. Per 'days' > 90, considera di passare a 'max' se supportato
            # o di usare /market_chart e inferire OHLC (meno preciso).
            # Attualmente, se 'days' > 90 per ohlc, l'API di CG dà dati orari, che dovremmo aggregare.
            # Per semplicità, limitiamo a 90 giorni o usiamo i giorni utente se minori.
            # Questo è un punto da rivedere per dati storici crypto estesi.
            if days_to_fetch_cg > 90: # Limite pratico per l'endpoint /ohlc se vogliamo dati giornalieri
                 st.warning(f"CoinGecko: Richiesta per {days_to_fetch_cg} giorni. L'endpoint OHLC potrebbe restituire dati orari o essere limitato. Si procede con max 90 giorni per dati giornalieri o i giorni selezionati se inferiori.")
                 days_to_fetch_cg = min(user_selected_days if user_selected_days > 0 else 90, 90)


            st.session_state.ss_raw_data_df = get_crypto_data(
                st.session_state.ss_ticker_input,
                vs_currency=CONFIG.get('coingecko',{}).get('vs_currency', 'usd'),
                days=days_to_fetch_cg 
            )
        
        if st.session_state.ss_raw_data_df is not None and not st.session_state.ss_raw_data_df.empty:
            start_dt_filter = pd.to_datetime(st.session_state.ss_start_date)
            end_dt_filter = pd.to_datetime(st.session_state.ss_end_date)
            # Assicura che l'indice sia DatetimeIndex prima di filtrare
            if not isinstance(st.session_state.ss_raw_data_df.index, pd.DatetimeIndex):
                st.session_state.ss_raw_data_df.index = pd.to_datetime(st.session_state.ss_raw_data_df.index)

            st.session_state.ss_raw_data_df = st.session_state.ss_raw_data_df[
                (st.session_state.ss_raw_data_df.index >= start_dt_filter) & 
                (st.session_state.ss_raw_data_df.index <= end_dt_filter)
            ]
            if st.session_state.ss_raw_data_df.empty:
                st.warning(f"Nessun dato per {st.session_state.ss_ticker_input} trovato nell'intervallo di date specificato ({st.session_state.ss_start_date.strftime('%Y-%m-%d')} - {st.session_state.ss_end_date.strftime('%Y-%m-%d')}).")
                st.session_state.ss_raw_data_df = None 
            else:
                st.success(f"Dati grezzi per {st.session_state.ss_ticker_input} caricati e filtrati. Shape: {st.session_state.ss_raw_data_df.shape}")
        elif st.session_state.ss_raw_data_df is None: 
            st.error(f"Fallimento nel caricamento dei dati grezzi per {st.session_state.ss_ticker_input}.")
        
        if st.session_state.ss_raw_data_df is not None and not st.session_state.ss_raw_data_df.empty:
            progress_bar.progress(25, text="Calcolo feature tecniche...")
            st.session_state.ss_features_df = calculate_technical_features(st.session_state.ss_raw_data_df)
            if st.session_state.ss_features_df.empty:
                st.error("Fallimento nel calcolo delle feature tecniche.")
            else:
                st.success(f"Feature tecniche calcolate. Shape: {st.session_state.ss_features_df.shape}")

                progress_bar.progress(40, text="Creazione target di predizione...")
                pred_horizon = CONFIG.get('ml_model', {}).get('prediction_target_horizon_days', 3)
                df_with_target = create_prediction_targets(st.session_state.ss_features_df, horizon=pred_horizon)
                target_col_name = f'target_{pred_horizon}d_pct_change'

                feature_cols_for_ml = CONFIG.get('ml_model',{}).get('feature_columns_for_training', ['MA20', 'MA50', 'RSI', 'StochRSI', 'Momentum'])
                feature_cols_for_ml = [col for col in feature_cols_for_ml if col in df_with_target.columns]

                if not feature_cols_for_ml:
                    st.error("Nessuna colonna feature valida trovata per il training/predizione ML.")
                elif target_col_name not in df_with_target.columns:
                    st.error(f"Colonna target '{target_col_name}' non creata o mancante per il training ML.")
                else:
                    predictions_series = None # Inizializza
                    if CONFIG.get('ml_model', {}).get('use_google_ai_studio', False):
                        progress_bar.progress(55, text="Ottenimento predizioni da Google AI Studio...")
                        st.warning("Integrazione Google AI Studio non completamente implementata.")
                        predictions_series = pd.Series(index=df_with_target.index, dtype=float) 
                    else: 
                        progress_bar.progress(55, text="Training modello RandomForest locale...")
                        st.session_state.ss_trained_ml_model = train_random_forest_model(
                            df_with_target, 
                            feature_columns=feature_cols_for_ml,
                            target_column=target_col_name,
                            n_estimators=CONFIG.get('ml_model',{}).get('random_forest_n_estimators', 100)
                        )
                        if st.session_state.ss_trained_ml_model:
                            st.success("Modello RandomForest addestrato.")
                            progress_bar.progress(70, text="Generazione predizioni ML...")
                            predictions_series = generate_model_predictions(
                                st.session_state.ss_trained_ml_model,
                                df_with_target, 
                                feature_columns=feature_cols_for_ml
                            )
                        else:
                            st.error("Fallimento training modello RandomForest.")
                            
                    if predictions_series is not None:
                        st.session_state.ss_target_and_preds_df = df_with_target.copy()
                        prediction_col_ml_name = f'prediction_{pred_horizon}d_pct_change'
                        st.session_state.ss_target_and_preds_df[prediction_col_ml_name] = predictions_series
                        st.success(f"Predizioni ML generate e aggiunte come '{prediction_col_ml_name}'.")
                    else:
                        st.error("Fallimento nella generazione delle predizioni ML.")

                if st.session_state.ss_target_and_preds_df is not None and prediction_col_ml_name in st.session_state.ss_target_and_preds_df:
                    progress_bar.progress(85, text="Generazione segnali di trading...")
                    df_ml_signals_only = generate_signals_from_ml_predictions(
                        st.session_state.ss_target_and_preds_df,
                        prediction_column_name=prediction_col_ml_name,
                        buy_threshold=CONFIG.get('signal_logic',{}).get('buy_threshold_change', 0.005),
                        sell_threshold=CONFIG.get('signal_logic',{}).get('sell_threshold_change', -0.005)
                    )
                    df_breakout_signals_only = detect_breakout_signals(
                        st.session_state.ss_features_df, 
                        high_low_period=CONFIG.get('signal_logic',{}).get('breakout_days_high_low_period', 20),
                        volume_avg_factor=CONFIG.get('signal_logic',{}).get('breakout_volume_avg_factor', 1.0),
                        volume_period=CONFIG.get('signal_logic',{}).get('breakout_days_high_low_period', 20) 
                    )
                    st.session_state.ss_final_signals_df = combine_signals(df_ml_signals_only, df_breakout_signals_only)
                    st.session_state.ss_final_signals_df = apply_trading_spreads(
                        st.session_state.ss_final_signals_df,
                        st.session_state.ss_asset_type,
                        CONFIG.get('spreads', {})
                    )
                    st.success("Segnali di trading finali generati.")

                    if not st.session_state.ss_final_signals_df.empty:
                        last_signal_row = st.session_state.ss_final_signals_df.iloc[-1]
                        st.session_state.ss_last_generated_signal_info = {
                            "ticker": st.session_state.ss_ticker_input,
                            "date": last_signal_row.name.strftime('%Y-%m-%d %H:%M:%S') if isinstance(last_signal_row.name, pd.Timestamp) else str(last_signal_row.name),
                            "ml_signal": last_signal_row.get('ml_signal', 'N/A'),
                            "breakout_signal": last_signal_row.get('breakout_signal', 'N/A'),
                            "close_price": f"{last_signal_row.get('Close', 0.0):.2f}" if 'Close' in last_signal_row else "N/A"
                        }
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
            
            progress_bar.progress(100, text="Analisi completata!")
            import time # Importa time qui se non già fatto globalmente
            time.sleep(1) 
            progress_bar.empty() 

    # Resetta il flag per evitare riesecuzione automatica al prossimo re-run di Streamlit dovuto a interazioni UI
    # Questo deve essere fatto fuori dal blocco 'with log_container' se vogliamo che la logica di visualizzazione sotto
    # si basi sullo stato attuale senza rieseguire tutta la pipeline
    if st.session_state.ss_analysis_run_triggered: # Controlla di nuovo perché potrebbe essere cambiato dentro il with
        st.session_state.ss_analysis_run_triggered = False


# --- AREA PRINCIPALE PER VISUALIZZAZIONE RISULTATI ---
st.markdown("---")
st.header(f"📊 Risultati per: {st.session_state.ss_ticker_input if st.session_state.ss_ticker_input else 'N/D'}")

if st.session_state.ss_final_signals_df is not None and not st.session_state.ss_final_signals_df.empty:
    if st.session_state.ss_last_generated_signal_info:
        st.subheader("📢 Ultimo Segnale Generato:")
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
    if st.session_state.ss_features_df is not None and not st.session_state.ss_features_df.empty:
        chart_fig = create_main_stock_chart(
            df_ohlcv_ma=st.session_state.ss_features_df, 
            df_signals=st.session_state.ss_final_signals_df,
            ticker=st.session_state.ss_ticker_input,
            ma_periods_to_show=CONFIG.get('visualization',{}).get('show_ma_periods', [20, 50])
        )
        st.plotly_chart(chart_fig, use_container_width=True)
    else:
        st.warning("Dati per il grafico (feature DF) non disponibili.")


    with st.expander("👁️ Visualizza Dati Tabellari Dettagliati (ultimi 100 record)"):
        if st.session_state.ss_raw_data_df is not None:
            st.markdown("#### Dati Grezzi (OHLCV)")
            st.dataframe(st.session_state.ss_raw_data_df.tail(100))
        if st.session_state.ss_features_df is not None:
            st.markdown("#### Feature Tecniche Calcolate")
            st.dataframe(st.session_state.ss_features_df.tail(100))
        if st.session_state.ss_target_and_preds_df is not None:
            st.markdown("#### Target di Predizione e Predizioni ML")
            st.dataframe(st.session_state.ss_target_and_preds_df.tail(100))
        if st.session_state.ss_final_signals_df is not None:
            st.markdown("#### Segnali Finali (ML e Breakout)")
            st.dataframe(st.session_state.ss_final_signals_df.tail(100))

elif st.session_state.get('ss_ticker_input'): 
    # Questo blocco viene raggiunto se ss_final_signals_df è None o vuoto, ma un ticker è stato inserito
    # e l'analisi potrebbe essere stata tentata (o è il primo caricamento con un ticker di default).
    if 'ss_analysis_run_triggered' not in st.session_state or not st.session_state.ss_analysis_run_triggered : # Se non è in corso un'analisi
        if st.session_state.get('ss_raw_data_df') is None and st.session_state.get('ss_ticker_input'):
             st.warning("Dati non ancora caricati o analisi fallita. Controlla i log di processo sopra se hai eseguito un'analisi.")
        elif st.session_state.get('ss_raw_data_df') is not None and st.session_state.get('ss_raw_data_df').empty :
             st.warning("Nessun dato grezzo disponibile per il ticker e l'intervallo selezionato.")

else: 
    st.info("👋 Benvenuto! Inserisci i parametri nella sidebar a sinistra e clicca 'Analizza e Genera Segnali' per iniziare.")

st.markdown("---")
st.caption(f"Dashboard v{APP_VERSION_FROM_CONFIG}. Controlla i messaggi DEBUG e gli errori nel log di processo per dettagli sull'esecuzione.")
st.caption(f"Ultimo aggiornamento dell'interfaccia Streamlit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}")

with st.sidebar.expander("🔍 DEBUG: Stato Sessione Completo", expanded=False):
    # Converti DataFrame in JSON rappresentabile (es. to_dict) per evitare errori con oggetti non serializzabili
    session_state_dict_for_json = {}
    for k, v in st.session_state.to_dict().items():
        if isinstance(v, pd.DataFrame):
            session_state_dict_for_json[k] = f"DataFrame with shape {v.shape}" if v is not None else "None"
        elif isinstance(v, (datetime, pd.Timestamp)):
             session_state_dict_for_json[k] = v.isoformat()
        # Aggiungere altre conversioni se necessario per altri tipi non JSON serializzabili
        else:
            session_state_dict_for_json[k] = v
    st.json(session_state_dict_for_json, expanded=False)
