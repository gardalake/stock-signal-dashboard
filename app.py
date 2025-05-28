# app.py - v1.6.5 (Main Application Orchestrator - Fix AttributeError, cache prep)
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import yaml 
import os 
import time 
import json # Importa il modulo json standard per il debug dello stato sessione

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
yaml_error_message_for_later = None # Inizializza

try:
    with open(CONFIG_FILE, 'r') as f:
        CONFIG = yaml.safe_load(f)
    APP_VERSION_FROM_CONFIG = CONFIG.get('version', 'v1.6.5-spec-impl (config fallback)')
    config_loaded_successfully_flag = True
except FileNotFoundError:
    APP_VERSION_FROM_CONFIG = "ERRORE CONFIG - File non trovato" 
    pass 
except yaml.YAMLError as e:
    APP_VERSION_FROM_CONFIG = "ERRORE CONFIG - YAML invalido"
    yaml_error_message_for_later = e 
    pass

# --- CONFIGURAZIONE PAGINA STREAMLIT ---
st.set_page_config(
    layout="wide",
    page_title=f"Stock Signal Dashboard {APP_VERSION_FROM_CONFIG}", 
    page_icon="📈" 
)

if config_loaded_successfully_flag:
    print(f"INFO [app.py_module]: {CONFIG_FILE} caricato. Versione: {APP_VERSION_FROM_CONFIG}")
    if 'config_loaded_successfully' not in st.session_state: 
        st.session_state.config_loaded_successfully = True
elif yaml_error_message_for_later is not None:
    st.error(f"ERRORE CRITICO nel parsing di '{CONFIG_FILE}': {yaml_error_message_for_later}. L'app non può continuare.")
    st.session_state.config_loaded_successfully = False
    st.stop()
else: # FileNotFoundError
    st.error(f"ERRORE CRITICO: '{CONFIG_FILE}' non trovato. L'app non può continuare.")
    st.session_state.config_loaded_successfully = False
    st.stop()

# --- GESTIONE CHIAVI API ---
ALPHA_VANTAGE_API_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY")
# ... (resto della gestione chiavi API come prima) ...
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

# --- STATO DELLA SESSIONE ---
default_session_state_values = {
    'ss_ticker_input': "AAPL", 
    'ss_asset_type': "stock", 
    'ss_start_date_display': datetime.now().date() - timedelta(days=90), 
    'ss_end_date_display': datetime.now().date(),     
    'ss_raw_data_df_full_history': None, 
    'ss_raw_data_df_filtered': None,   
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

# --- TITOLO E HEADER ---
st.title(f"📈 Stock & Crypto Signal Dashboard")
st.markdown(f"**Versione:** `{APP_VERSION_FROM_CONFIG}` - _Basato sulle specifiche del progetto._")
st.markdown("---")

# --- SIDEBAR ---
with st.sidebar:
    # ... (contenuto della sidebar come prima) ...
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
        st.session_state.ss_ticker_input = "AAPL" if st.session_state.ss_asset_type == "stock" else "bitcoin" # Resetta il ticker di default
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
        st.stop() 

    st.markdown("---") 
    if st.button("🚀 Analizza e Genera Segnali", type="primary", use_container_width=True, key="run_analysis_button"):
        if not st.session_state.ss_ticker_input:
            st.warning("Per favore, inserisci un simbolo Ticker o Coin ID valido.")
        else:
            st.session_state.ss_analysis_run_triggered = True 
            st.session_state.ss_raw_data_df_full_history = None 
            st.session_state.ss_raw_data_df_filtered = None   
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
        # ... (pipeline di analisi come prima, non la ripeto tutta per brevità ma è identica a quella dell'ultimo app.py che ti ho dato) ...
        progress_bar = st.progress(0, text="Inizio analisi...")
        MIN_DAYS_HISTORY_FOR_ML = CONFIG.get('ml_model', {}).get('min_days_for_indicators_and_training', 200)
        actual_data_load_start_date = st.session_state.ss_end_date_display - timedelta(days=MIN_DAYS_HISTORY_FOR_ML -1) 

        progress_bar.progress(10, text=f"Caricamento storico esteso per {st.session_state.ss_ticker_input}...")
        if st.session_state.ss_asset_type == "stock":
            # ... (codice caricamento stock) ...
            if not ALPHA_VANTAGE_API_KEY:
                st.error("Impossibile caricare dati azioni: Chiave API Alpha Vantage mancante.")
                st.session_state.ss_raw_data_df_full_history = None 
            else:
                st.session_state.ss_raw_data_df_full_history = get_stock_data(
                    ALPHA_VANTAGE_API_KEY, 
                    st.session_state.ss_ticker_input,
                    CONFIG.get('alpha_vantage', {}).get('function', 'TIME_SERIES_DAILY_ADJUSTED'),
                    "full" 
                )
        elif st.session_state.ss_asset_type == "crypto":
            # ... (codice caricamento crypto) ...
            days_to_fetch_for_coingecko = (datetime.now().date() - actual_data_load_start_date).days + 1
            if days_to_fetch_for_coingecko <=0: days_to_fetch_for_coingecko = MIN_DAYS_HISTORY_FOR_ML 
            st.write(f"DEBUG [app.py]: Caricamento crypto - Data inizio richiesta API: {actual_data_load_start_date}, Giorni da fetchare: {days_to_fetch_for_coingecko}")
            st.session_state.ss_raw_data_df_full_history = get_crypto_data(
                st.session_state.ss_ticker_input,
                vs_currency=CONFIG.get('coingecko',{}).get('vs_currency', 'usd'),
                days=days_to_fetch_for_coingecko 
            )
        
        if st.session_state.ss_raw_data_df_full_history is not None and not st.session_state.ss_raw_data_df_full_history.empty:
            # ... (codice filtraggio e processamento ML come prima) ...
            start_dt_display_filter = pd.to_datetime(st.session_state.ss_start_date_display)
            end_dt_display_filter = pd.to_datetime(st.session_state.ss_end_date_display)
            if not isinstance(st.session_state.ss_raw_data_df_full_history.index, pd.DatetimeIndex):
                st.session_state.ss_raw_data_df_full_history.index = pd.to_datetime(st.session_state.ss_raw_data_df_full_history.index)
            st.session_state.ss_raw_data_df_filtered = st.session_state.ss_raw_data_df_full_history[
                (st.session_state.ss_raw_data_df_full_history.index >= start_dt_display_filter) & 
                (st.session_state.ss_raw_data_df_full_history.index <= end_dt_display_filter)
            ].copy() 
            if st.session_state.ss_raw_data_df_filtered.empty:
                st.warning(f"Nessun dato per {st.session_state.ss_ticker_input} nell'intervallo visualizzazione.")
            else:
                st.success(f"Storico caricato. Dati per visualizzazione: Shape {st.session_state.ss_raw_data_df_filtered.shape}")
        
        elif st.session_state.ss_raw_data_df_full_history is None: 
            st.error(f"Fallimento caricamento storico per {st.session_state.ss_ticker_input}.")
        
        if st.session_state.ss_raw_data_df_full_history is not None and not st.session_state.ss_raw_data_df_full_history.empty:
            if len(st.session_state.ss_raw_data_df_full_history) < MIN_DAYS_HISTORY_FOR_ML / 2: 
                st.warning(f"Storico ({len(st.session_state.ss_raw_data_df_full_history)} righe) potrebbe essere insufficiente per ML (min: {MIN_DAYS_HISTORY_FOR_ML}).")
            
            progress_bar.progress(25, text="Calcolo feature tecniche...")
            st.session_state.ss_features_df = calculate_technical_features(st.session_state.ss_raw_data_df_full_history)
            
            if st.session_state.ss_features_df.empty or len(st.session_state.ss_features_df) < 10: 
                st.error("Fallimento calcolo feature o dati insufficienti.")
            else:
                st.success(f"Feature calcolate. Shape: {st.session_state.ss_features_df.shape}")
                progress_bar.progress(40, text="Creazione target predizione...")
                pred_horizon = CONFIG.get('ml_model', {}).get('prediction_target_horizon_days', 3)
                df_with_target_full_hist = create_prediction_targets(st.session_state.ss_features_df, horizon=pred_horizon) 
                target_col_name = f'target_{pred_horizon}d_pct_change'
                feature_cols_for_ml = CONFIG.get('ml_model',{}).get('feature_columns_for_training', ['MA20', 'MA50', 'RSI', 'StochRSI_K', 'Momentum_ROC10', 'ADX', 'MACD_line'])
                feature_cols_for_ml = [col for col in feature_cols_for_ml if col in df_with_target_full_hist.columns]

                if not feature_cols_for_ml:
                    st.error("Nessuna feature valida per training ML.")
                elif target_col_name not in df_with_target_full_hist.columns:
                    st.error(f"Target '{target_col_name}' mancante per training ML.")
                else:
                    predictions_series = None 
                    if CONFIG.get('ml_model', {}).get('use_google_ai_studio', False):
                        st.warning("Integrazione Google AI Studio non implementata.")
                        predictions_series = pd.Series(index=df_with_target_full_hist.index, dtype=float) 
                    else: 
                        progress_bar.progress(55, text="Training RandomForest...")
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
                                df_with_target_full_hist, 
                                feature_columns=feature_cols_for_ml
                            )
                        else:
                            st.error("Fallimento training RandomForest.")
                            
                    if predictions_series is not None:
                        st.session_state.ss_target_and_preds_df = df_with_target_full_hist.copy()
                        prediction_col_ml_name = f'prediction_{pred_horizon}d_pct_change'
                        st.session_state.ss_target_and_preds_df[prediction_col_ml_name] = predictions_series
                        st.success(f"Predizioni ML generate come '{prediction_col_ml_name}'.")
                    else:
                        st.error("Fallimento generazione predizioni ML.")

                if st.session_state.ss_target_and_preds_df is not None and prediction_col_ml_name in st.session_state.ss_target_and_preds_df:
                    progress_bar.progress(85, text="Generazione segnali trading...")
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
                    df_signals_full_history = combine_signals(df_ml_signals_only, df_breakout_signals_only)
                    df_signals_full_history = apply_trading_spreads(
                        df_signals_full_history,
                        st.session_state.ss_asset_type,
                        CONFIG.get('spreads', {})
                    )
                    
                    if not df_signals_full_history.empty and st.session_state.ss_raw_data_df_filtered is not None and not st.session_state.ss_raw_data_df_filtered.empty:
                        # Filtra per l'indice di ss_raw_data_df_filtered se esiste e non è vuoto
                        common_index = st.session_state.ss_raw_data_df_filtered.index.intersection(df_signals_full_history.index)
                        if not common_index.empty:
                            st.session_state.ss_final_signals_df = df_signals_full_history.loc[common_index].copy()
                            st.success(f"Segnali finali filtrati per visualizzazione. Shape: {st.session_state.ss_final_signals_df.shape}")
                        else:
                            st.warning("Nessun indice comune tra segnali e dati filtrati per visualizzazione.")
                            st.session_state.ss_final_signals_df = pd.DataFrame()
                    elif st.session_state.ss_raw_data_df_filtered is None or st.session_state.ss_raw_data_df_filtered.empty:
                        st.warning("Intervallo visualizzazione senza dati, segnali non mostrati per questo intervallo.")
                        st.session_state.ss_final_signals_df = pd.DataFrame() 
                    else: # Caso fallback, mostra tutto se il filtro sopra non va a buon fine
                        st.session_state.ss_final_signals_df = df_signals_full_history.copy()


                    if st.session_state.ss_final_signals_df is not None and not st.session_state.ss_final_signals_df.empty:
                        last_signal_row = st.session_state.ss_final_signals_df.iloc[-1]
                        # ... (logica ultimo segnale, suoni, email come prima) ...
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
                    st.warning("Impossibile generare segnali: predizioni ML non disponibili.")
        else: 
             st.error("Elaborazione ML interrotta: storico grezzo non caricato.")
            
        progress_bar.progress(100, text="Analisi completata!")
        time.sleep(0.5) # Ridotto sleep
        progress_bar.empty() 

    if st.session_state.ss_analysis_run_triggered: 
        st.session_state.ss_analysis_run_triggered = False

# --- AREA PRINCIPALE PER VISUALIZZAZIONE RISULTATI ---
# ... (sezione visualizzazione come prima, assicurati che i DataFrame usati siano corretti) ...
st.markdown("---")
st.header(f"📊 Risultati per: {st.session_state.ss_ticker_input if st.session_state.ss_ticker_input else 'N/D'}")

if st.session_state.ss_final_signals_df is not None and not st.session_state.ss_final_signals_df.empty:
    if st.session_state.ss_last_generated_signal_info:
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
    
    # Per il grafico, usiamo ss_features_df MA filtrato per l'intervallo di visualizzazione (ss_raw_data_df_filtered.index)
    # e ss_final_signals_df (che dovrebbe essere già filtrato per lo stesso intervallo)
    df_features_for_chart_display = pd.DataFrame() # Inizializza vuoto
    if st.session_state.ss_features_df is not None and not st.session_state.ss_features_df.empty and \
       st.session_state.ss_raw_data_df_filtered is not None and not st.session_state.ss_raw_data_df_filtered.empty:
        
        common_idx_chart = st.session_state.ss_raw_data_df_filtered.index.intersection(st.session_state.ss_features_df.index)
        if not common_idx_chart.empty:
            df_features_for_chart_display = st.session_state.ss_features_df.loc[common_idx_chart].copy()
        else:
            st.warning("Nessun indice comune tra dati filtrati e feature per il grafico.")

    if not df_features_for_chart_display.empty:
        chart_fig = create_main_stock_chart(
            df_ohlcv_ma=df_features_for_chart_display, 
            df_signals=st.session_state.ss_final_signals_df, # Dovrebbe essere già filtrato
            ticker=st.session_state.ss_ticker_input,
            ma_periods_to_show=CONFIG.get('visualization',{}).get('show_ma_periods', [20, 50])
        )
        st.plotly_chart(chart_fig, use_container_width=True)
    else:
        st.warning("Dati insufficienti o non allineati per visualizzare il grafico principale.")


    with st.expander("👁️ Visualizza Dati Tabellari Dettagliati (ultimi 100 record dell'intervallo visualizzato)"):
        # ... (visualizzazione tabelle come prima, usando _filtered e _final_signals) ...
        if st.session_state.ss_raw_data_df_filtered is not None: 
            st.markdown("#### Dati Grezzi (OHLCV - Intervallo Visualizzato)")
            st.dataframe(st.session_state.ss_raw_data_df_filtered.tail(100))
        
        if st.session_state.ss_features_df is not None and st.session_state.ss_raw_data_df_filtered is not None and not st.session_state.ss_raw_data_df_filtered.empty:
            common_idx_feat_tbl = st.session_state.ss_raw_data_df_filtered.index.intersection(st.session_state.ss_features_df.index)
            if not common_idx_feat_tbl.empty:
                 st.markdown("#### Feature Tecniche Calcolate (Intervallo Visualizzato)")
                 st.dataframe(st.session_state.ss_features_df.loc[common_idx_feat_tbl].tail(100))
        
        if st.session_state.ss_target_and_preds_df is not None and st.session_state.ss_raw_data_df_filtered is not None and not st.session_state.ss_raw_data_df_filtered.empty:
            common_idx_pred_tbl = st.session_state.ss_raw_data_df_filtered.index.intersection(st.session_state.ss_target_and_preds_df.index)
            if not common_idx_pred_tbl.empty:
                st.markdown("#### Target di Predizione e Predizioni ML (Intervallo Visualizzato)")
                st.dataframe(st.session_state.ss_target_and_preds_df.loc[common_idx_pred_tbl].tail(100))
        
        if st.session_state.ss_final_signals_df is not None: 
            st.markdown("#### Segnali Finali (ML e Breakout - Intervallo Visualizzato)")
            st.dataframe(st.session_state.ss_final_signals_df.tail(100))


elif st.session_state.get('ss_ticker_input'): 
    if 'ss_analysis_run_triggered' not in st.session_state or not st.session_state.ss_analysis_run_triggered :
        if st.session_state.get('ss_raw_data_df_full_history') is None and st.session_state.get('ss_ticker_input'): 
             st.warning("Dati non ancora caricati o analisi fallita. Controlla i log di processo sopra se hai eseguito un'analisi.")
        elif st.session_state.get('ss_raw_data_df_filtered') is not None and st.session_state.get('ss_raw_data_df_filtered').empty : 
             st.warning("Nessun dato grezzo disponibile per il ticker e l'intervallo di visualizzazione selezionato.")
else: 
    st.info("👋 Benvenuto! Inserisci i parametri nella sidebar a sinistra e clicca 'Analizza e Genera Segnali' per iniziare.")

st.markdown("---")
st.caption(f"Dashboard v{APP_VERSION_FROM_CONFIG}. Controlla i messaggi DEBUG e gli errori nel log di processo per dettagli sull'esecuzione.")
st.caption(f"Ultimo aggiornamento dell'interfaccia Streamlit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}")

with st.sidebar.expander("🔍 DEBUG: Stato Sessione Completo", expanded=False):
    session_state_dict_for_json = {}
    for k, v in st.session_state.to_dict().items():
        if isinstance(v, pd.DataFrame):
            session_state_dict_for_json[k] = f"DataFrame with shape {v.shape}" if v is not None else "None"
        elif isinstance(v, (datetime, pd.Timestamp, pd.Period)):
             session_state_dict_for_json[k] = str(v)
        else:
            try: 
                json.dumps(v) 
                session_state_dict_for_json[k] = v
            except (TypeError, OverflowError): 
                session_state_dict_for_json[k] = str(v) 
    st.json(session_state_dict_for_json, expanded=False)
