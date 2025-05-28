# app.py - v1.6.6 (UI Refactor - Top Controls, Predefined Selections)
import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta # Aggiunto date
import yaml 
import os 
import time 
import json 

from logger_utils import setup_logger
logger = setup_logger(__name__) 

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
    APP_VERSION_FROM_CONFIG = CONFIG.get('version', 'v1.6.6-ui-refactor (config fallback)') 
    config_loaded_successfully_flag = True
except FileNotFoundError:
    APP_VERSION_FROM_CONFIG = "ERRORE CONFIG - File non trovato" 
    print(f"CRITICAL_ERROR [app.py_module]: {CONFIG_FILE} non trovato.")
except yaml.YAMLError as e:
    APP_VERSION_FROM_CONFIG = "ERRORE CONFIG - YAML invalido"
    yaml_error_message_for_later = e 
    print(f"CRITICAL_ERROR [app.py_module]: Errore parsing {CONFIG_FILE}: {e}")

# --- CONFIGURAZIONE PAGINA STREAMLIT ---
st.set_page_config(
    layout="wide",
    page_title=f"Stock Signal Dashboard {APP_VERSION_FROM_CONFIG}", 
    page_icon="📊" 
)

if config_loaded_successfully_flag:
    logger.info(f"{CONFIG_FILE} caricato. Versione da config: {APP_VERSION_FROM_CONFIG}")
    if 'config_loaded_successfully' not in st.session_state: 
        st.session_state.config_loaded_successfully = True
elif yaml_error_message_for_later is not None:
    st.error(f"ERRORE CRITICO nel parsing di '{CONFIG_FILE}': {yaml_error_message_for_later}. L'app non può continuare.")
    logger.critical(f"Errore parsing '{CONFIG_FILE}': {yaml_error_message_for_later}")
    st.session_state.config_loaded_successfully = False
    st.stop()
else: 
    st.error(f"ERRORE CRITICO: '{CONFIG_FILE}' non trovato. L'app non può continuare.")
    logger.critical(f"'{CONFIG_FILE}' non trovato.")
    st.session_state.config_loaded_successfully = False
    st.stop()

# --- GESTIONE CHIAVI API ---
ALPHA_VANTAGE_API_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY")
# I warning/error per le chiavi API verranno mostrati più avanti, se necessario
# ... (codice gestione altre chiavi API come prima) ...
GOOGLE_AI_STUDIO_URL = None 
GOOGLE_AI_STUDIO_TOKEN = None
EMAIL_SMTP_PASSWORD = None 
LOADED_SECRETS = { 
    "ALPHA_VANTAGE_API_KEY": ALPHA_VANTAGE_API_KEY,
    CONFIG.get('ml_model', {}).get('google_ai_studio_url_secret_name', 'GOOGLE_AI_STUDIO_URL_PLACEHOLDER'): GOOGLE_AI_STUDIO_URL,
    CONFIG.get('ml_model', {}).get('google_ai_studio_token_secret_name', 'GOOGLE_AI_STUDIO_TOKEN_PLACEHOLDER'): GOOGLE_AI_STUDIO_TOKEN,
    CONFIG.get('email_notifications', {}).get('smtp_password_secret_name', 'EMAIL_SMTP_PASSWORD_PLACEHOLDER'): EMAIL_SMTP_PASSWORD
}

# --- STATO DELLA SESSIONE ---
default_session_state_values = {
    'ss_selected_asset_type': "stock",  
    'ss_selected_symbol': "AAPL",       
    'ss_selected_interval_label': "Giornaliero (ultimi 3 mesi)", # Etichetta per UI
    
    'ss_data_ohlcv_full': None,         
    'ss_data_ohlcv_display': None,      
    'ss_features_full': None,           
    'ss_target_and_preds_full': None,   
    'ss_final_signals_display': None,   
    'ss_trained_ml_model': None,     
    'ss_last_signal_info_display': None, 
    'ss_analysis_run_flag': False       
}
for key, value in default_session_state_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- DEFINIZIONI PER UI CONTROLS ---
AVAILABLE_STOCK_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "NONEXISTENT_STOCK"]
AVAILABLE_CRYPTO_COINS = ["bitcoin", "ethereum", "dogecoin", "NONEXISTENT_CRYPTO"] 

# Intervalli: Etichetta Utente -> (codice_interno, av_interval_str, av_function_str, cg_days_for_granularity, is_intraday_flag)
# av_interval_str: per TIME_SERIES_INTRADAY (es. "60min")
# av_function_str: la funzione AV da chiamare (es. TIME_SERIES_INTRADAY, TIME_SERIES_DAILY_ADJUSTED)
# cg_days_for_granularity: quanti giorni di storico chiedere a CoinGecko per ottenere la granularità desiderata
# is_intraday_flag: True se l'intervallo primario è intraday
AVAILABLE_INTERVALS_MAP = {
    "1 Ora (ultime 24h)":    ("1H",   "60min", "TIME_SERIES_INTRADAY", 1,    True), 
    "4 Ore (ultimi 7gg)":    ("4H",   "60min", "TIME_SERIES_INTRADAY", 7,    True), # AV 60min, poi aggregare a 4H
    "Giornaliero (ultimi 3m)": ("1D_3M","Daily", CONFIG.get('alpha_vantage', {}).get('function', 'TIME_SERIES_DAILY_ADJUSTED'), 90,   False),
    "Giornaliero (ultima sett)": ("1D_1W","Daily", CONFIG.get('alpha_vantage', {}).get('function', 'TIME_SERIES_DAILY_ADJUSTED'), 7,    False),
    "Giornaliero (ultimo mese)": ("1D_1M","Daily", CONFIG.get('alpha_vantage', {}).get('function', 'TIME_SERIES_DAILY_ADJUSTED'), 30,   False),
    "Giornaliero (ultimo anno)": ("1D_1Y","Daily", CONFIG.get('alpha_vantage', {}).get('function', 'TIME_SERIES_DAILY_ADJUSTED'), 365,  False),
}
# Inizializza il simbolo di default se cambia il tipo di asset
if 'prev_asset_type_ui' not in st.session_state or st.session_state.prev_asset_type_ui != st.session_state.ss_selected_asset_type:
    st.session_state.ss_selected_symbol = AVAILABLE_STOCK_SYMBOLS[0] if st.session_state.ss_selected_asset_type == "stock" else AVAILABLE_CRYPTO_COINS[0]
    st.session_state.prev_asset_type_ui = st.session_state.ss_selected_asset_type

# --- LAYOUT CONTROLLI UI (IN ALTO) ---
st.title(f"📈 Stock & Crypto Signal Dashboard")
st.markdown(f"**Versione:** `{APP_VERSION_FROM_CONFIG}`")
st.markdown("---")

# Gestione warning chiave API
api_warning_placeholder = st.empty()
if not ALPHA_VANTAGE_API_KEY and CONFIG.get('alpha_vantage') and st.session_state.ss_selected_asset_type == "stock": 
    api_warning_placeholder.warning("Chiave API Alpha Vantage non configurata. I dati per le azioni non saranno disponibili.")
else:
    api_warning_placeholder.empty()


cols_ui = st.columns([0.25, 0.35, 0.25, 0.15]) # Proporzioni aggiustate

with cols_ui[0]: # Tipo Asset + Simbolo
    st.session_state.ss_selected_asset_type = st.radio(
        "Tipo Asset:", options=["stock", "crypto"],
        index=["stock", "crypto"].index(st.session_state.ss_selected_asset_type),
        horizontal=True, key="ui_asset_type_top"
    )
    current_symbols_list = AVAILABLE_STOCK_SYMBOLS if st.session_state.ss_selected_asset_type == "stock" else AVAILABLE_CRYPTO_COINS
    if st.session_state.ss_selected_symbol not in current_symbols_list: # Assicura che il simbolo sia valido
        st.session_state.ss_selected_symbol = current_symbols_list[0]
    st.session_state.ss_selected_symbol = st.selectbox(
        "Simbolo:", options=current_symbols_list,
        index=current_symbols_list.index(st.session_state.ss_selected_symbol),
        key="ui_symbol_select_top"
    )

with cols_ui[1]: # Intervallo/Granularità
    st.session_state.ss_selected_interval_label = st.selectbox(
        "Intervallo/Granularità Dati:",
        options=list(AVAILABLE_INTERVALS_MAP.keys()),
        index=list(AVAILABLE_INTERVALS_MAP.keys()).index(st.session_state.ss_selected_interval_label),
        key="ui_interval_select_top",
        help="Seleziona la granularità dei dati e l'orizzonte di visualizzazione."
    )

# cols_ui[2] potrebbe essere usato per date custom se reintrodotte
# with cols_ui[2]:
# st.text("Placeholder data custom")

with cols_ui[3]: # Bottone Analizza
    st.markdown("<br>", unsafe_allow_html=True) # Allineamento verticale
    if st.button("📊 Analizza", type="primary", use_container_width=True, key="ui_analyze_button_top"):
        if not st.session_state.ss_selected_symbol:
            st.warning("Seleziona un simbolo.")
            logger.warning("Analisi richiesta senza simbolo.")
        else:
            st.session_state.ss_analysis_run_flag = True
            # Reset stati
            st.session_state.ss_data_ohlcv_full = None
            st.session_state.ss_data_ohlcv_display = None
            st.session_state.ss_features_full = None
            st.session_state.ss_target_and_preds_full = None
            st.session_state.ss_final_signals_display = None
            st.session_state.ss_trained_ml_model = None
            st.session_state.ss_last_signal_info_display = None
            logger.info(f"Analisi avviata per {st.session_state.ss_selected_symbol}, intervallo: {st.session_state.ss_selected_interval_label}")
            # st.experimental_rerun() # Potrebbe essere utile

st.markdown("---")

# --- LOGICA DI CALCOLO START/END DATE PER API E DISPLAY ---
# Questo blocco determina le date effettive in base alla selezione dell'utente
# e i parametri per le chiamate API.

interval_code, av_api_interval, av_api_function, cg_api_days_granularity, interval_is_intraday = AVAILABLE_INTERVALS_MAP[st.session_state.ss_selected_interval_label]

# Determina le date di fine e inizio per la VISUALIZZAZIONE
# L'end_date è sempre "oggi" (o l'ultimo giorno con dati).
# La start_date dipende dall'intervallo scelto.
_display_end_date = date.today() # Usiamo date, non datetime, per gli input dell'utente
if interval_code == "1H":
    _display_start_date = _display_end_date - timedelta(days=1) # Ultime 24h
elif interval_code == "4H":
    _display_start_date = _display_end_date - timedelta(days=7) # Ultimi 7gg
elif interval_code == "1D_3M":
    _display_start_date = _display_end_date - timedelta(days=90)
elif interval_code == "1D_1W":
    _display_start_date = _display_end_date - timedelta(weeks=1)
elif interval_code == "1D_1M":
    _display_start_date = _display_end_date - timedelta(days=30) # Approssimato
elif interval_code == "1D_1Y":
    _display_start_date = _display_end_date - timedelta(days=365)
else: # Fallback (dovrebbe essere coperto da selezione valida)
    _display_start_date = _display_end_date - timedelta(days=90)

# Determina i parametri per il CARICAMENTO DATI (potrebbe essere uno storico più lungo per ML)
MIN_DAYS_FOR_ML_AND_TA = CONFIG.get('ml_model', {}).get('min_days_for_indicators_and_training', 200)

# La data di inizio per caricare lo storico completo per ML deve precedere _display_start_date
# di abbastanza giorni da coprire le finestre degli indicatori e il training.
_api_data_load_start_date = _display_start_date - timedelta(days=MIN_DAYS_FOR_ML_AND_TA)

# Parametri specifici per le API
_av_outputsize_param = "full" # Per daily. Per intraday, 'compact' è spesso meglio se 'full' è troppo.
if interval_is_intraday:
    # Per AlphaVantage intraday, 'full' può dare fino a 1-2 mesi, ma è pesante.
    # 'compact' dà gli ultimi 100 punti. Scegliamo 'compact' per alleggerire.
    # Se servono più dati intraday, si deve gestire la paginazione o usare un piano AV a pagamento.
    _av_outputsize_param = "compact" if st.session_state.ss_selected_asset_type == "stock" else "full"


# Per CoinGecko, `days` è quanti giorni indietro da oggi.
# Assicuriamoci di caricare abbastanza per coprire da _api_data_load_start_date a oggi.
_cg_days_to_fetch_param = (date.today() - _api_data_load_start_date).days +1
if _cg_days_to_fetch_param <= 0: # Fallback
    _cg_days_to_fetch_param = MIN_DAYS_FOR_ML_AND_TA 
# CoinGecko per intraday: se chiedi pochi giorni (es. cg_api_days_granularity), ti dà dati più fini.
# Se l'intervallo selezionato è intraday, usiamo il numero di giorni ottimale per quella granularità.
if interval_is_intraday and st.session_state.ss_selected_asset_type == "crypto":
    _cg_days_to_fetch_param = cg_api_days_granularity 


# --- PIPELINE DI ELABORAZIONE DATI E SEGNALI ---
if st.session_state.ss_analysis_run_flag:
    log_container = st.container()
    with log_container:
        st.markdown("### ⚙️ Log di Processo dell'Analisi")
        progress_bar = st.progress(0, text="Inizio analisi...")
        logger.info(f"Inizio caricamento dati. Display: {_display_start_date} a {_display_end_date}. API load start: {_api_data_load_start_date}")

        # 1. CARICAMENTO DATI STORICO COMPLETO
        progress_bar.progress(10, text=f"Caricamento storico per {st.session_state.ss_selected_symbol}...")
        if st.session_state.ss_selected_asset_type == "stock":
            if not ALPHA_VANTAGE_API_KEY:
                st.error("Chiave API Alpha Vantage mancante.")
                logger.error("Chiave API AV mancante.")
                st.session_state.ss_data_ohlcv_full = None
            else:
                st.session_state.ss_data_ohlcv_full = get_stock_data(
                    ALPHA_VANTAGE_API_KEY, 
                    st.session_state.ss_selected_symbol,
                    av_api_function, # Funzione basata sull'intervallo
                    _av_outputsize_param, # Outputsize basato su intervallo
                    # Aggiungi av_api_interval se la funzione è intraday
                    **(dict(interval=av_api_interval) if av_api_function == "TIME_SERIES_INTRADAY" else {})
                )
        elif st.session_state.ss_selected_asset_type == "crypto":
            logger.debug(f"Caricamento crypto - Giorni da fetchare: {_cg_days_to_fetch_param}")
            st.session_state.ss_data_ohlcv_full = get_crypto_data(
                st.session_state.ss_selected_symbol,
                vs_currency=CONFIG.get('coingecko',{}).get('vs_currency', 'usd'),
                days=_cg_days_to_fetch_param,
                # Aggiungi interval se l'API lo supporta per granularità fine
                # Per ora, get_crypto_data gestisce questo internamente
            )

        # 2. FILTRAGGIO DATI PER VISUALIZZAZIONE
        if st.session_state.ss_data_ohlcv_full is not None and not st.session_state.ss_data_ohlcv_full.empty:
            logger.info(f"Storico completo caricato. Shape: {st.session_state.ss_data_ohlcv_full.shape}")
            _start_dt_display_filter = pd.to_datetime(_display_start_date) # Converti date a datetime
            _end_dt_display_filter = pd.to_datetime(_display_end_date)
            
            if not isinstance(st.session_state.ss_data_ohlcv_full.index, pd.DatetimeIndex):
                st.session_state.ss_data_ohlcv_full.index = pd.to_datetime(st.session_state.ss_data_ohlcv_full.index)

            # Se l'intervallo è intraday, il filtraggio delle date potrebbe essere più complesso
            # Per ora, filtriamo sull'indice normalizzato se i dati sono orari/minuti
            # e vogliamo visualizzare un range di giorni.
            df_to_filter = st.session_state.ss_data_ohlcv_full
            if interval_is_intraday: # Per dati intraday, l'indice ha anche l'ora
                 st.session_state.ss_data_ohlcv_display = df_to_filter[
                    (df_to_filter.index >= _start_dt_display_filter) & 
                    (df_to_filter.index <= _end_dt_display_filter + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)) # Includi tutto il giorno di fine
                ].copy()
            else: # Per dati giornalieri
                st.session_state.ss_data_ohlcv_display = df_to_filter[
                    (df_to_filter.index.normalize() >= _start_dt_display_filter) & # Normalizza indice per confronto date
                    (df_to_filter.index.normalize() <= _end_dt_display_filter)
                ].copy()


            if st.session_state.ss_data_ohlcv_display.empty:
                st.warning(f"Nessun dato per '{st.session_state.ss_selected_symbol}' nell'intervallo di visualizzazione ({_display_start_date.strftime('%Y-%m-%d')} - {_display_end_date.strftime('%Y-%m-%d')}).")
                logger.warning(f"Dati filtrati per display vuoti per {st.session_state.ss_selected_symbol}.")
            else:
                st.success(f"Dati per visualizzazione pronti. Shape: {st.session_state.ss_data_ohlcv_display.shape}")
        
        elif st.session_state.ss_data_ohlcv_full is None: 
            st.error(f"Fallimento nel caricamento dello storico completo per {st.session_state.ss_selected_symbol}.")
            logger.error(f"Fallimento caricamento storico completo per {st.session_state.ss_selected_symbol}.")

        # --- INIZIO ELABORAZIONE ML (su ss_data_ohlcv_full) ---
        # ... (resto della pipeline ML e segnali, usando ss_data_ohlcv_full per calcoli e ss_data_ohlcv_display per filtraggio finale) ...
        # Assicurati che i DataFrame passati alle funzioni successive (calculate_features, etc.) siano quelli corretti.
        # Se ss_data_ohlcv_full è None, salta questa parte.
        
        if st.session_state.ss_data_ohlcv_full is not None and not st.session_state.ss_data_ohlcv_full.empty:
            # Verifica se lo storico caricato è sufficiente
            if len(st.session_state.ss_data_ohlcv_full) < MIN_DAYS_FOR_ML_AND_TA / 2: # Soglia arbitraria
                st.warning(f"Storico caricato ({len(st.session_state.ss_data_ohlcv_full)} punti) potrebbe essere insufficiente per analisi ML robusta (minimo suggerito: {MIN_DAYS_FOR_ML_AND_TA}).")
                logger.warning(f"Storico per ML ({len(st.session_state.ss_data_ohlcv_full)} punti) potrebbe essere insufficiente.")
            
            progress_bar.progress(25, text="Calcolo feature tecniche...")
            st.session_state.ss_features_full = calculate_technical_features(st.session_state.ss_data_ohlcv_full)
            
            if st.session_state.ss_features_full.empty or len(st.session_state.ss_features_full) < 10: 
                st.error("Fallimento calcolo feature o dati insufficienti.")
                logger.error("Fallimento calcolo feature o dati post-feature insufficienti.")
            else:
                st.success(f"Feature calcolate su storico. Shape: {st.session_state.ss_features_full.shape}")
                logger.info(f"Feature calcolate su storico. Shape: {st.session_state.ss_features_full.shape}")

                progress_bar.progress(40, text="Creazione target predizione...")
                pred_horizon = CONFIG.get('ml_model', {}).get('prediction_target_horizon_days', 3)
                # Per dati intraday, l'orizzonte di predizione in "giorni" deve essere tradotto in numero di periodi
                # Esempio: se dati orari (24 periodi/giorno), 3 giorni = 72 periodi.
                # Questa logica va aggiunta se si usano dati intraday per il training ML.
                # Per ora, assumiamo che il training ML avvenga su dati aggregati giornalieri o che l'orizzonte sia in periodi.
                # Se interval_is_intraday, pred_horizon potrebbe significare "periodi" della granularità corrente.
                # Questa è una semplificazione per ora.
                
                df_with_target_full = create_prediction_targets(st.session_state.ss_features_full, horizon=pred_horizon) 
                target_col_name = f'target_{pred_horizon}d_pct_change' # O _periods_ se intraday

                # ... (resto della pipeline ML e segnali, usando df_with_target_full, ss_features_full) ...
                # ... e poi filtrando il risultato finale (df_signals_full_history_combined) sull'indice di ss_data_ohlcv_display ...

                # Esempio di come continuare (molto abbreviato):
                feature_cols_ml_config = CONFIG.get('ml_model',{}).get('feature_columns_for_training', ['MA20', 'MA50', 'RSI', 'StochRSI_K', 'Momentum_ROC10', 'ADX', 'MACD_line'])
                feature_cols_for_ml = [col for col in feature_cols_ml_config if col in df_with_target_full.columns]

                if feature_cols_for_ml and target_col_name in df_with_target_full.columns:
                    logger.info(f"Inizio training RandomForest. Feature: {feature_cols_for_ml}")
                    st.session_state.ss_trained_ml_model = train_random_forest_model(df_with_target_full, feature_cols_for_ml, target_col_name)
                    if st.session_state.ss_trained_ml_model:
                        predictions_series = generate_model_predictions(st.session_state.ss_trained_ml_model, df_with_target_full, feature_cols_for_ml)
                        if predictions_series is not None:
                            st.session_state.ss_target_and_preds_full = df_with_target_full.copy()
                            st.session_state.ss_target_and_preds_full[f'prediction_{pred_horizon}d_pct_change'] = predictions_series
                            
                            # Genera segnali su dati completi
                            df_ml_signals_full = generate_signals_from_ml_predictions(st.session_state.ss_target_and_preds_full, f'prediction_{pred_horizon}d_pct_change', CONFIG.get('signal_logic',{}).get('buy_threshold_change', 0.005), CONFIG.get('signal_logic',{}).get('sell_threshold_change', -0.005))
                            df_breakout_full = detect_breakout_signals(st.session_state.ss_features_full) # Passare parametri da config
                            df_signals_combined_full = combine_signals(df_ml_signals_full, df_breakout_full)
                            df_signals_combined_full = apply_trading_spreads(df_signals_combined_full, st.session_state.ss_selected_asset_type, CONFIG.get('spreads',{}))

                            # Filtra per display
                            if st.session_state.ss_data_ohlcv_display is not None and not st.session_state.ss_data_ohlcv_display.empty:
                                common_idx_disp = st.session_state.ss_data_ohlcv_display.index.intersection(df_signals_combined_full.index)
                                if not common_idx_disp.empty:
                                    st.session_state.ss_final_signals_display = df_signals_combined_full.loc[common_idx_disp].copy()
                                    st.success(f"Segnali finali filtrati per display. Shape: {st.session_state.ss_final_signals_display.shape}")
                                    logger.info(f"Segnali finali filtrati. Shape: {st.session_state.ss_final_signals_display.shape}")

                                    if not st.session_state.ss_final_signals_display.empty:
                                        last_sig_row_disp = st.session_state.ss_final_signals_display.iloc[-1]
                                        # ... (logica ultimo segnale come prima)
                                        st.session_state.ss_last_signal_info_display = {
                                            "ticker": st.session_state.ss_selected_symbol,
                                            "date": last_sig_row_disp.name.strftime('%Y-%m-%d %H:%M:%S') if isinstance(last_sig_row_disp.name, pd.Timestamp) else str(last_sig_row_disp.name),
                                            "ml_signal": last_sig_row_disp.get('ml_signal', 'N/A'),
                                            "breakout_signal": last_sig_row_disp.get('breakout_signal', 'N/A'),
                                            "close_price": f"{last_sig_row_disp.get('Close', 0.0):.2f}" if 'Close' in last_sig_row_disp else "N/A"
                                        }
                                        # ... (suoni/email)
                                else:
                                    st.warning("Nessun segnale comune con l'intervallo di display.")
                                    logger.warning("Nessun segnale comune con intervallo display.")
                            else:
                                st.warning("Nessun dato nell'intervallo di display per mostrare i segnali.")
                                logger.warning("Dati display vuoti, nessun segnale da mostrare.")
                        else:
                            st.error("Fallimento generazione predizioni ML.")
                            logger.error("Fallimento predizioni ML.")
                    else:
                        st.error("Fallimento training modello.")
                        logger.error("Fallimento training modello.")
                else: # feature_cols_for_ml vuote o target mancante
                    st.error("Pipeline ML interrotta: feature o target mancanti.")
                    logger.error("Pipeline ML interrotta: feature o target mancanti.")
        else: 
             st.error("Elaborazione ML interrotta: storico grezzo non caricato o vuoto.")
             logger.error("Elaborazione ML interrotta: dati grezzi storici non disponibili.")
            
        progress_bar.progress(100, text="Analisi completata!")
        time.sleep(0.5) 
        progress_bar.empty() 

    if st.session_state.get('ss_analysis_run_flag', False): 
        st.session_state.ss_analysis_run_flag = False
        logger.debug("Flag ss_analysis_run_flag resettato a False.")


# --- AREA PRINCIPALE PER VISUALIZZAZIONE RISULTATI ---
st.markdown("---")
st.header(f"📊 Risultati per: {st.session_state.ss_selected_symbol if st.session_state.ss_selected_symbol else 'N/D'}")

if st.session_state.ss_final_signals_display is not None and not st.session_state.ss_final_signals_display.empty:
    if st.session_state.ss_last_signal_info_display:
        # ... (visualizzazione ultimo segnale come prima) ...
        st.subheader("📢 Ultimo Segnale Generato (nell'intervallo visualizzato):")
        sig_info = st.session_state.ss_last_signal_info_display
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
    if st.session_state.ss_features_full is not None and not st.session_state.ss_features_full.empty and \
       st.session_state.ss_data_ohlcv_display is not None and not st.session_state.ss_data_ohlcv_display.empty:
        
        common_idx_chart = st.session_state.ss_data_ohlcv_display.index.intersection(st.session_state.ss_features_full.index)
        if not common_idx_chart.empty:
            df_features_for_chart_display = st.session_state.ss_features_full.loc[common_idx_chart].copy()
            logger.debug(f"DataFrame per grafico (df_features_for_chart_display) creato. Shape: {df_features_for_chart_display.shape}")
        else:
            st.warning("Nessun indice comune tra dati filtrati e feature per il grafico.")
            logger.warning("Nessun indice comune per df_features_for_chart_display.")

    if not df_features_for_chart_display.empty:
        chart_fig = create_main_stock_chart(
            df_ohlcv_ma=df_features_for_chart_display, 
            df_signals=st.session_state.ss_final_signals_display, 
            ticker=st.session_state.ss_selected_symbol,
            ma_periods_to_show=CONFIG.get('visualization',{}).get('show_ma_periods', [20, 50])
        )
        st.plotly_chart(chart_fig, use_container_width=True)
    else:
        if st.session_state.get('ss_data_ohlcv_full') is not None: 
             st.warning("Dati insufficienti o non allineati per visualizzare il grafico principale.")
             logger.warning("Dati insufficienti/non allineati per grafico principale.")

    with st.expander("👁️ Visualizza Dati Tabellari Dettagliati (ultimi 100 record dell'intervallo visualizzato)"):
        # Mostra ss_data_ohlcv_display
        if st.session_state.ss_data_ohlcv_display is not None and not st.session_state.ss_data_ohlcv_display.empty: 
            st.markdown("#### Dati Grezzi (OHLCV - Intervallo Visualizzato)")
            st.dataframe(st.session_state.ss_data_ohlcv_display.tail(100))
        
        # Filtra e mostra ss_features_full
        if st.session_state.ss_features_full is not None and \
           st.session_state.ss_data_ohlcv_display is not None and not st.session_state.ss_data_ohlcv_display.empty:
            common_idx_feat_tbl = st.session_state.ss_data_ohlcv_display.index.intersection(st.session_state.ss_features_full.index)
            if not common_idx_feat_tbl.empty:
                 st.markdown("#### Feature Tecniche Calcolate (Intervallo Visualizzato)")
                 st.dataframe(st.session_state.ss_features_full.loc[common_idx_feat_tbl].tail(100))
        
        # Filtra e mostra ss_target_and_preds_full
        if st.session_state.ss_target_and_preds_full is not None and \
           st.session_state.ss_data_ohlcv_display is not None and not st.session_state.ss_data_ohlcv_display.empty:
            common_idx_pred_tbl = st.session_state.ss_data_ohlcv_display.index.intersection(st.session_state.ss_target_and_preds_full.index)
            if not common_idx_pred_tbl.empty:
                st.markdown("#### Target di Predizione e Predizioni ML (Intervallo Visualizzato)")
                st.dataframe(st.session_state.ss_target_and_preds_full.loc[common_idx_pred_tbl].tail(100))
        
        # ss_final_signals_display è già filtrato
        if st.session_state.ss_final_signals_display is not None and not st.session_state.ss_final_signals_display.empty: 
            st.markdown("#### Segnali Finali (ML e Breakout - Intervallo Visualizzato)")
            st.dataframe(st.session_state.ss_final_signals_display.tail(100))

elif st.session_state.get('ss_selected_symbol'): 
    if 'ss_analysis_run_flag' not in st.session_state or not st.session_state.ss_analysis_run_flag : 
        if st.session_state.get('ss_data_ohlcv_full') is None and st.session_state.get('ss_selected_symbol'): 
             st.warning("Dati non ancora caricati o analisi fallita. Controlla i log di processo sopra se hai eseguito un'analisi.")
        elif st.session_state.get('ss_data_ohlcv_display') is not None and st.session_state.get('ss_data_ohlcv_display').empty : 
             st.warning("Nessun dato grezzo disponibile per il ticker e l'intervallo di visualizzazione selezionato.")
else: 
    st.info("👋 Benvenuto! Inserisci i parametri nei controlli in alto e clicca 'Analizza' per iniziare.")

st.markdown("---")
st.caption(f"Dashboard v{APP_VERSION_FROM_CONFIG}. Controlla i log del server per dettagli DEBUG/INFO.")
st.caption(f"Ultimo aggiornamento dell'interfaccia Streamlit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}")

with st.expander("🔍 DEBUG: Stato Sessione Completo (in Sidebar)", expanded=False): # Spostato in fondo alla pagina, non sidebar
    session_state_dict_for_json = {}
    for k, v_item in st.session_state.to_dict().items(): 
        if isinstance(v_item, pd.DataFrame):
            session_state_dict_for_json[k] = f"DataFrame with shape {v_item.shape}" if v_item is not None else "None"
        elif isinstance(v_item, (datetime, date, pd.Timestamp, pd.Period)): # Aggiunto 'date'
             session_state_dict_for_json[k] = str(v_item)
        else:
            try: 
                json.dumps(v_item) 
                session_state_dict_for_json[k] = v_item
            except (TypeError, OverflowError): 
                session_state_dict_for_json[k] = str(v_item) 
    st.json(session_state_dict_for_json, expanded=False)
