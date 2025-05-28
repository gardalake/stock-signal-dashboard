# app.py - v1.6.7 (UI Refactor - Fix NameError for default session state)
import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta 
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
    APP_VERSION_FROM_CONFIG = CONFIG.get('version', 'v1.6.7-table-ui (config fallback)') 
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
    page_title=f"Asset Signal Dashboard {APP_VERSION_FROM_CONFIG}", 
    page_icon="📊" 
)

if config_loaded_successfully_flag:
    logger.info(f"{CONFIG_FILE} caricato. Versione da config: {APP_VERSION_FROM_CONFIG}")
    if 'config_loaded_successfully' not in st.session_state: 
        st.session_state.config_loaded_successfully = True
elif yaml_error_message_for_later is not None:
    st.error(f"ERRORE CRITICO nel parsing di '{CONFIG_FILE}': {yaml_error_message_for_later}.")
    st.stop()
else: 
    st.error(f"ERRORE CRITICO: '{CONFIG_FILE}' non trovato.")
    st.stop()

# --- GESTIONE CHIAVI API ---
ALPHA_VANTAGE_API_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY")
GOOGLE_AI_STUDIO_URL = None 
GOOGLE_AI_STUDIO_TOKEN = None
EMAIL_SMTP_PASSWORD = None 
google_url_secret_key = CONFIG.get('ml_model', {}).get('google_ai_studio_url_secret_name', 'GOOGLE_AI_STUDIO_URL') 
google_token_secret_key = CONFIG.get('ml_model', {}).get('google_ai_studio_token_secret_name', 'GOOGLE_AI_STUDIO_TOKEN')
email_pwd_secret_key = CONFIG.get('email_notifications', {}).get('smtp_password_secret_name', 'EMAIL_SMTP_PASSWORD')
if CONFIG.get('ml_model', {}).get('use_google_ai_studio', False):
    GOOGLE_AI_STUDIO_URL = st.secrets.get(google_url_secret_key)
    GOOGLE_AI_STUDIO_TOKEN = st.secrets.get(google_token_secret_key)
    if not GOOGLE_AI_STUDIO_URL or not GOOGLE_AI_STUDIO_TOKEN:
        logger.warning(f"Google AI Studio: URL ({google_url_secret_key}) o Token ({google_token_secret_key}) mancanti.")
if CONFIG.get('email_notifications', {}).get('enabled', False):
    EMAIL_SMTP_PASSWORD = st.secrets.get(email_pwd_secret_key)
    if not EMAIL_SMTP_PASSWORD:
        logger.warning(f"Email: password SMTP ({email_pwd_secret_key}) non in st.secrets.")
LOADED_SECRETS = { 
    "ALPHA_VANTAGE_API_KEY": ALPHA_VANTAGE_API_KEY,
    google_url_secret_key: GOOGLE_AI_STUDIO_URL,
    google_token_secret_key: GOOGLE_AI_STUDIO_TOKEN,
    email_pwd_secret_key: EMAIL_SMTP_PASSWORD
}

# --- DEFINIZIONI ASSET E INTERVALLI (SPOSTATE QUI PRIMA DELLO STATO SESSIONE) ---
TARGET_ASSETS_LIST = [
    {"name": "Apple Inc.", "symbol": "AAPL", "type": "stock", "cg_id": None, "last_price": "N/A", "ml_signal": "N/A", "breakout_signal": "N/A"},
    {"name": "Microsoft Corp.", "symbol": "MSFT", "type": "stock", "cg_id": None, "last_price": "N/A", "ml_signal": "N/A", "breakout_signal": "N/A"},
    {"name": "Google (Alphabet)", "symbol": "GOOGL", "type": "stock", "cg_id": None, "last_price": "N/A", "ml_signal": "N/A", "breakout_signal": "N/A"},
    {"name": "Amazon.com Inc.", "symbol": "AMZN", "type": "stock", "cg_id": None, "last_price": "N/A", "ml_signal": "N/A", "breakout_signal": "N/A"},
    {"name": "NVIDIA Corp.", "symbol": "NVDA", "type": "stock", "cg_id": None, "last_price": "N/A", "ml_signal": "N/A", "breakout_signal": "N/A"},
    {"name": "Tesla Inc.", "symbol": "TSLA", "type": "stock", "cg_id": None, "last_price": "N/A", "ml_signal": "N/A", "breakout_signal": "N/A"},
    {"name": "Meta Platforms", "symbol": "META", "type": "stock", "cg_id": None, "last_price": "N/A", "ml_signal": "N/A", "breakout_signal": "N/A"},
    {"name": "Bitcoin", "symbol": "BTC", "type": "crypto", "cg_id": "bitcoin", "last_price": "N/A", "ml_signal": "N/A", "breakout_signal": "N/A"},
    {"name": "Ethereum", "symbol": "ETH", "type": "crypto", "cg_id": "ethereum", "last_price": "N/A", "ml_signal": "N/A", "breakout_signal": "N/A"},
    {"name": "Solana", "symbol": "SOL", "type": "crypto", "cg_id": "solana", "last_price": "N/A", "ml_signal": "N/A", "breakout_signal": "N/A"},
]
AVAILABLE_STOCK_SYMBOLS = [asset["symbol"] for asset in TARGET_ASSETS_LIST if asset["type"] == "stock"]
AVAILABLE_CRYPTO_COINS_INFO = {asset["symbol"]: asset["cg_id"] for asset in TARGET_ASSETS_LIST if asset["type"] == "crypto"}
AVAILABLE_CRYPTO_SYMBOLS = list(AVAILABLE_CRYPTO_COINS_INFO.keys())

AVAILABLE_INTERVALS_MAP = {
    "1 Ora (ultime 24h)":    ("1H",   "60min", "TIME_SERIES_INTRADAY", 1,    True, 24), # Display 24 ore 
    "4 Ore (ultimi 7gg)":    ("4H",   "60min", "TIME_SERIES_INTRADAY", 7,    True, 7*6), # Display 7gg * 6 periodi da 4h
    "Giornaliero (ultimi 3m)": ("1D_3M","Daily", CONFIG.get('alpha_vantage', {}).get('function', 'TIME_SERIES_DAILY_ADJUSTED'), 90,   False,90), 
    "Giornaliero (ultima sett)": ("1D_1W","Daily", CONFIG.get('alpha_vantage', {}).get('function', 'TIME_SERIES_DAILY_ADJUSTED'), 7,    False,7), 
    "Giornaliero (ultimo mese)": ("1D_1M","Daily", CONFIG.get('alpha_vantage', {}).get('function', 'TIME_SERIES_DAILY_ADJUSTED'), 30,   False,30), 
    "Giornaliero (ultimo anno)": ("1D_1Y","Daily", CONFIG.get('alpha_vantage', {}).get('function', 'TIME_SERIES_DAILY_ADJUSTED'), 365,  False,365),
}
DEFAULT_INTERVAL_LABEL = "Giornaliero (ultimi 3m)" 

# --- STATO DELLA SESSIONE ---
default_session_state_values = {
    'ss_selected_asset_type': "stock",  
    'ss_selected_symbol': AVAILABLE_STOCK_SYMBOLS[0] if AVAILABLE_STOCK_SYMBOLS else "N/A", # Usa il primo stock, se esiste       
    'ss_selected_interval_label': DEFAULT_INTERVAL_LABEL, 
    'ss_data_ohlcv_full': None, 'ss_data_ohlcv_display': None,      
    'ss_features_full': None, 'ss_target_and_preds_full': None,   
    'ss_final_signals_display': None, 'ss_trained_ml_model': None,     
    'ss_last_signal_info_display': None, 'ss_analysis_run_flag': False       
}
for key, value in default_session_state_values.items():
    if key not in st.session_state: st.session_state[key] = value

if 'prev_asset_type_ui' not in st.session_state or st.session_state.prev_asset_type_ui != st.session_state.ss_selected_asset_type:
    st.session_state.ss_selected_symbol = AVAILABLE_STOCK_SYMBOLS[0] if st.session_state.ss_selected_asset_type == "stock" and AVAILABLE_STOCK_SYMBOLS else (AVAILABLE_CRYPTO_SYMBOLS[0] if AVAILABLE_CRYPTO_SYMBOLS else "N/A")
    st.session_state.prev_asset_type_ui = st.session_state.ss_selected_asset_type

# --- LAYOUT CONTROLLI UI ---
st.title(f"📊 Asset Signal Dashboard")
st.caption(f"Versione: {APP_VERSION_FROM_CONFIG}")
st.markdown("---")

api_warning_placeholder_main = st.empty()
if st.session_state.ss_selected_asset_type == "stock" and not ALPHA_VANTAGE_API_KEY and CONFIG.get('alpha_vantage'): 
    api_warning_placeholder_main.warning("Chiave API Alpha Vantage non configurata. Dati per azioni non disponibili.")
else:
    api_warning_placeholder_main.empty()

cols_ui = st.columns([0.25, 0.35, 0.25, 0.15]) 
with cols_ui[0]: 
    st.session_state.ss_selected_asset_type = st.radio(
        "Tipo Asset:", options=["stock", "crypto"],
        index=["stock", "crypto"].index(st.session_state.ss_selected_asset_type),
        horizontal=True, key="ui_asset_type_top"
    )
    current_symbols_list_ui = AVAILABLE_STOCK_SYMBOLS if st.session_state.ss_selected_asset_type == "stock" else AVAILABLE_CRYPTO_SYMBOLS
    if not current_symbols_list_ui: # Se la lista è vuota (non dovrebbe succedere con i default)
        st.session_state.ss_selected_symbol = "N/A"
        st.selectbox("Simbolo:", options=["Nessun simbolo disponibile"], index=0, key="ui_symbol_select_top", disabled=True)
    else:
        if st.session_state.ss_selected_symbol not in current_symbols_list_ui: 
            st.session_state.ss_selected_symbol = current_symbols_list_ui[0]
        st.session_state.ss_selected_symbol = st.selectbox("Simbolo:", options=current_symbols_list_ui, key="ui_symbol_select_top")

with cols_ui[1]: 
    st.session_state.ss_selected_interval_label = st.selectbox(
        "Intervallo/Granularità Dati:", options=list(AVAILABLE_INTERVALS_MAP.keys()),
        key="ui_interval_select_top", help="Seleziona la granularità e l'orizzonte di visualizzazione."
    )

with cols_ui[3]: 
    st.markdown("<br>", unsafe_allow_html=True) 
    if st.button("📊 Analizza", type="primary", use_container_width=True, key="ui_analyze_button_top"):
        if not st.session_state.ss_selected_symbol or st.session_state.ss_selected_symbol == "N/A": # Modificato controllo
            st.warning("Seleziona un simbolo valido.")
        else:
            st.session_state.ss_analysis_run_flag = True
            # Reset stati ...
            st.session_state.ss_data_ohlcv_full = None; st.session_state.ss_data_ohlcv_display = None
            st.session_state.ss_features_full = None; st.session_state.ss_target_and_preds_full = None
            st.session_state.ss_final_signals_display = None; st.session_state.ss_trained_ml_model = None
            st.session_state.ss_last_signal_info_display = None
            logger.info(f"Analisi avviata per {st.session_state.ss_selected_symbol}, intervallo: {st.session_state.ss_selected_interval_label}")

st.markdown("---")

# --- LOGICA CALCOLO DATE E PARAMETRI API ---
interval_details_tuple = AVAILABLE_INTERVALS_MAP.get(st.session_state.ss_selected_interval_label)
if not interval_details_tuple:
    logger.error(f"Dettagli intervallo non validi: {st.session_state.ss_selected_interval_label}. Uso default.")
    interval_details_tuple = AVAILABLE_INTERVALS_MAP[DEFAULT_INTERVAL_LABEL] 
interval_code, av_api_interval, av_api_function, cg_api_days_granularity, interval_is_intraday, display_units_value = interval_details_tuple

_display_end_date_dt = datetime.now() 
if interval_code == "1H": _display_start_date_dt = _display_end_date_dt - timedelta(hours=display_units_value)
elif interval_code == "4H": _display_start_date_dt = _display_end_date_dt - timedelta(hours=display_units_value * 4)
elif interval_code.startswith("1D"): 
    num_part = interval_code.split('_')[-1][:-1] # Estrae il numero, es. '3' da '1D_3M'
    unit_part = interval_code.split('_')[-1][-1] # Estrae l'unità, es. 'M' da '1D_3M'
    num_val = int(num_part) if num_part.isdigit() else display_units_value # Usa display_units_value se non c'è numero specifico
    if unit_part == "W": _display_start_date_dt = _display_end_date_dt - timedelta(weeks=num_val)
    elif unit_part == "M": _display_start_date_dt = _display_end_date_dt - timedelta(days=num_val * 30) 
    elif unit_part == "Y": _display_start_date_dt = _display_end_date_dt - timedelta(days=num_val * 365)
    else: _display_start_date_dt = _display_end_date_dt - timedelta(days=display_units_value) 
else: _display_start_date_dt = _display_end_date_dt - timedelta(days=30) 

MIN_DAYS_FOR_ML_AND_TA = CONFIG.get('ml_model', {}).get('min_days_for_indicators_and_training', 200)
# Usa .date() per sottrazioni di giorni da un oggetto date, poi riconverti se necessario
_api_data_load_start_date_dt_for_calc = _display_start_date_dt.date() - timedelta(days=MIN_DAYS_FOR_ML_AND_TA)

_av_outputsize_param = "compact" if interval_is_intraday and st.session_state.ss_selected_asset_type == "stock" else "full"
_cg_days_to_fetch_param = (date.today() - _api_data_load_start_date_dt_for_calc).days + 1
if _cg_days_to_fetch_param <= 0: _cg_days_to_fetch_param = MIN_DAYS_FOR_ML_AND_TA 
if interval_is_intraday and st.session_state.ss_selected_asset_type == "crypto":
    _cg_days_to_fetch_param = cg_api_days_granularity 

# --- PIPELINE DI ELABORAZIONE ---
if st.session_state.ss_analysis_run_flag:
    log_container = st.container()
    with log_container:
        # --- INIZIO BLOCCO PIPELINE ---
        st.markdown(f"### ⚙️ Analisi per: {st.session_state.ss_selected_symbol} - Intervallo: {st.session_state.ss_selected_interval_label.split('(')[0].strip()}")
        progress_bar = st.progress(0, text="Inizio analisi...")
        logger.info(f"Inizio caricamento dati. Display: {_display_start_date_dt.strftime('%Y-%m-%d %H:%M')} a {_display_end_date_dt.strftime('%Y-%m-%d %H:%M')}. API load start: {_api_data_load_start_date_dt_for_calc.strftime('%Y-%m-%d')}")

        # 1. CARICAMENTO DATI
        current_symbol_to_fetch = st.session_state.ss_selected_symbol
        current_asset_type_to_fetch = st.session_state.ss_selected_asset_type
        cg_id_to_fetch = None
        if current_asset_type_to_fetch == "crypto":
            cg_id_to_fetch = AVAILABLE_CRYPTO_COINS_INFO.get(current_symbol_to_fetch)
            if not cg_id_to_fetch:
                st.error(f"ID CoinGecko non trovato per il simbolo crypto: {current_symbol_to_fetch}")
                st.session_state.ss_data_ohlcv_full = None


        progress_bar.progress(10, text=f"Caricamento storico per {current_symbol_to_fetch}...")
        logger.info(f"Inizio caricamento dati per {current_symbol_to_fetch}.")
        if current_asset_type_to_fetch == "stock":
            if not ALPHA_VANTAGE_API_KEY:
                st.error("Chiave API Alpha Vantage mancante.")
                st.session_state.ss_data_ohlcv_full = None 
            else:
                av_call_params = {}
                if av_api_function == "TIME_SERIES_INTRADAY": av_call_params['av_interval'] = av_api_interval
                st.session_state.ss_data_ohlcv_full = get_stock_data(ALPHA_VANTAGE_API_KEY, current_symbol_to_fetch, av_api_function, _av_outputsize_param, **av_call_params)
        elif current_asset_type_to_fetch == "crypto" and cg_id_to_fetch:
            logger.debug(f"Caricamento crypto {cg_id_to_fetch} - Giorni: {_cg_days_to_fetch_param}, intervallo: {interval_code}")
            st.session_state.ss_data_ohlcv_full = get_crypto_data(cg_id_to_fetch, CONFIG.get('coingecko',{}).get('vs_currency', 'usd'), _cg_days_to_fetch_param, interval_code)
        
        # ... (resto della pipeline di analisi e visualizzazione come prima, assicurandoti che l'indentazione sia corretta)
        # Il blocco pipeline è lungo, quindi lo abbrevio qui per focalizzarci sulla correzione.
        # Assicurati che l'indentazione del blocco che inizia con:
        # if st.session_state.ss_data_ohlcv_full is not None and not st.session_state.ss_data_ohlcv_full.empty:
        # sia corretta e coerente.
        
        # Esempio abbreviato della continuazione della pipeline (ASSICURATI CHE L'INDENTAZIONE SIA CORRETTA NEL TUO FILE COMPLETO):
        if st.session_state.ss_data_ohlcv_full is not None and not st.session_state.ss_data_ohlcv_full.empty:
            # ... (logica filtraggio per ss_data_ohlcv_display) ...
            _start_dt_display_filter_pd = pd.to_datetime(_display_start_date_dt)
            _end_dt_display_filter_pd = pd.to_datetime(_display_end_date_dt)
            if not isinstance(st.session_state.ss_data_ohlcv_full.index, pd.DatetimeIndex): st.session_state.ss_data_ohlcv_full.index = pd.to_datetime(st.session_state.ss_data_ohlcv_full.index)
            df_to_filter = st.session_state.ss_data_ohlcv_full
            if interval_is_intraday: st.session_state.ss_data_ohlcv_display = df_to_filter[(df_to_filter.index >= _start_dt_display_filter_pd) & (df_to_filter.index < _end_dt_display_filter_pd + pd.Timedelta(days=1))].copy()
            else: st.session_state.ss_data_ohlcv_display = df_to_filter[(df_to_filter.index.normalize() >= _start_dt_display_filter_pd.normalize()) & (df_to_filter.index.normalize() <= _end_dt_display_filter_pd.normalize())].copy()
            if st.session_state.ss_data_ohlcv_display.empty: st.warning(f"Nessun dato per display dopo filtraggio.")
            else: st.success(f"Dati per display pronti. Shape: {st.session_state.ss_data_ohlcv_display.shape}")

            progress_bar.progress(25, text="Calcolo feature...")
            st.session_state.ss_features_full = calculate_technical_features(st.session_state.ss_data_ohlcv_full)
            if st.session_state.ss_features_full.empty or len(st.session_state.ss_features_full) < 10:
                st.error("Fallimento calcolo feature o dati insuff.")
            else:
                st.success(f"Feature calcolate. Shape: {st.session_state.ss_features_full.shape}")
                # ... (continua pipeline ML come nella versione precedente)
                pred_horizon = CONFIG.get('ml_model', {}).get('prediction_target_horizon_days', 3)
                df_with_target_full = create_prediction_targets(st.session_state.ss_features_full, horizon=pred_horizon)
                target_col_name = f'target_{pred_horizon}d_pct_change'
                feature_cols_ml_config = CONFIG.get('ml_model',{}).get('feature_columns_for_training', ['MA20', 'MA50', 'RSI'])
                feature_cols_for_ml = [col for col in feature_cols_ml_config if col in df_with_target_full.columns]

                if feature_cols_for_ml and target_col_name in df_with_target_full.columns:
                    st.session_state.ss_trained_ml_model = train_random_forest_model(df_with_target_full, feature_cols_for_ml, target_col_name, n_estimators=CONFIG.get('ml_model',{}).get('random_forest_n_estimators', 100))
                    if st.session_state.ss_trained_ml_model:
                        predictions_series = generate_model_predictions(st.session_state.ss_trained_ml_model, df_with_target_full, feature_cols_for_ml)
                        if predictions_series is not None:
                            st.session_state.ss_target_and_preds_full = df_with_target_full.copy()
                            prediction_col_name_ml = f'prediction_{pred_horizon}d_pct_change' # Usa un nome consistente
                            st.session_state.ss_target_and_preds_full[prediction_col_name_ml] = predictions_series
                            
                            df_ml_signals_full = generate_signals_from_ml_predictions(st.session_state.ss_target_and_preds_full, prediction_col_name_ml, CONFIG.get('signal_logic',{}).get('buy_threshold_change', 0.005), CONFIG.get('signal_logic',{}).get('sell_threshold_change', -0.005))
                            df_breakout_full = detect_breakout_signals(st.session_state.ss_features_full)
                            df_signals_combined_full = combine_signals(df_ml_signals_full, df_breakout_full)
                            df_signals_combined_full = apply_trading_spreads(df_signals_combined_full, st.session_state.ss_selected_asset_type, CONFIG.get('spreads',{}))

                            # Aggiorna tabella UI per l'asset CORRENTE
                            asset_sym_current = st.session_state.ss_selected_symbol
                            if not df_signals_combined_full.empty:
                                last_full_sig = df_signals_combined_full.iloc[-1]
                                if asset_sym_current in st.session_state.ss_asset_table_data:
                                    st.session_state.ss_asset_table_data[asset_sym_current]["ml_signal"] = last_full_sig.get('ml_signal', 'N/A')
                                    st.session_state.ss_asset_table_data[asset_sym_current]["breakout_signal"] = last_full_sig.get('breakout_signal', 'N/A')
                                    st.session_state.ss_asset_table_data[asset_sym_current]["last_price"] = f"{last_full_sig.get('Close', 0.0):.2f}" if 'Close' in last_full_sig else "N/A"
                            
                            # Filtra per display
                            if st.session_state.ss_data_ohlcv_display is not None and not st.session_state.ss_data_ohlcv_display.empty:
                                common_idx_disp = st.session_state.ss_data_ohlcv_display.index.intersection(df_signals_combined_full.index)
                                if not common_idx_disp.empty:
                                    st.session_state.ss_final_signals_display = df_signals_combined_full.loc[common_idx_disp].copy()
                                    st.success(f"Segnali filtrati per display. Shape: {st.session_state.ss_final_signals_display.shape}")
                                    if not st.session_state.ss_final_signals_display.empty:
                                        last_disp_sig = st.session_state.ss_final_signals_display.iloc[-1]
                                        st.session_state.ss_last_signal_info_display = {"ticker": st.session_state.ss_selected_symbol, "date": str(last_disp_sig.name), "ml_signal": last_disp_sig.get('ml_signal'), "breakout_signal": last_disp_sig.get('breakout_signal'), "close_price": f"{last_disp_sig.get('Close',0):.2f}"}
                                        # Suoni/Email
                                        if last_disp_sig.get('ml_signal') == 'BUY': play_buy_signal_sound(CONFIG.get('sound_utils',{}))
                                        elif last_disp_sig.get('ml_signal') == 'SELL': play_sell_signal_sound(CONFIG.get('sound_utils',{}))
        else: 
             st.error("Elaborazione ML interrotta: storico grezzo non caricato.")
        # --- FINE BLOCCO PIPELINE ---
        progress_bar.progress(100, text="Analisi completata!")
        time.sleep(0.5) 
        progress_bar.empty() 

    if st.session_state.get('ss_analysis_run_flag', False): 
        st.session_state.ss_analysis_run_flag = False
        logger.debug("Flag ss_analysis_run_flag resettato.")

# --- AREA PRINCIPALE PER VISUALIZZAZIONE RISULTATI ---
# ... (Sezione visualizzazione come prima, usa le variabili di stato corrette)
st.markdown("---")
# Usa le info dell'asset attualmente selezionato per il titolo dei risultati
# Se l'analisi è stata fatta, current_asset_info è definito dai bottoni, altrimenti usa il default.
asset_info_for_header = st.session_state.get('ss_current_asset_display_info')
if not asset_info_for_header: # Fallback se non è mai stato cliccato un bottone
    asset_info_for_header = {
        "name": TARGET_ASSETS_LIST[0]["name"] if TARGET_ASSETS_LIST else "N/A",
        "symbol": st.session_state.ss_selected_symbol,
        "interval_label_short": AVAILABLE_INTERVALS_MAP[DEFAULT_INTERVAL_LABEL][0] # Usa il codice intervallo
    }

st.header(f"📊 Risultati per: {asset_info_for_header.get('name')} ({asset_info_for_header.get('symbol')}) - {asset_info_for_header.get('interval_label_short')}")

if st.session_state.ss_final_signals_display is not None and not st.session_state.ss_final_signals_display.empty:
    # ... (visualizzazione ultimo segnale e grafico come prima)
    if st.session_state.ss_last_signal_info_display:
        st.subheader("📢 Ultimo Segnale (nell'intervallo visualizzato):")
        sig_info = st.session_state.ss_last_signal_info_display
        ml_color = "green" if sig_info['ml_signal'] == "BUY" else "red" if sig_info['ml_signal'] == "SELL" else "gray"
        breakout_color = "blue" if sig_info['breakout_signal'] == "BULLISH" else "orange" if sig_info['breakout_signal'] == "BEARISH" else "gray"
        st.markdown(f"""
        *   **Data:** `{sig_info['date']}` *   **Segnale ML:** <span style='color:{ml_color}; font-weight:bold;'>{sig_info['ml_signal']}</span>
        *   **Breakout:** <span style='color:{breakout_color};'>{sig_info['breakout_signal']}</span> *   **Prezzo Chiusura:** `{sig_info['close_price']}`
        """, unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("📈 Grafico Interattivo")
    df_features_for_chart = pd.DataFrame() 
    if st.session_state.ss_features_full is not None and not st.session_state.ss_features_full.empty and \
       st.session_state.ss_data_ohlcv_display is not None and not st.session_state.ss_data_ohlcv_display.empty:
        common_idx_chart = st.session_state.ss_data_ohlcv_display.index.intersection(st.session_state.ss_features_full.index)
        if not common_idx_chart.empty:
            df_features_for_chart = st.session_state.ss_features_full.loc[common_idx_chart].copy()
    if not df_features_for_chart.empty:
        chart_fig = create_main_stock_chart(df_features_for_chart, st.session_state.ss_final_signals_display, asset_info_for_header["symbol"], CONFIG.get('visualization',{}).get('show_ma_periods', [20, 50]))
        st.plotly_chart(chart_fig, use_container_width=True)
    else:
        if st.session_state.get('ss_data_ohlcv_full') is not None: st.warning("Dati grafico insuff.")
    with st.expander("👁️ Dati Tabellari (ultimi 100 record dell'intervallo)"): # ... tabelle
        if st.session_state.ss_data_ohlcv_display is not None and not st.session_state.ss_data_ohlcv_display.empty: st.dataframe(st.session_state.ss_data_ohlcv_display.tail(100))
        # ... altre tabelle filtrate
elif 'ss_analysis_run_flag' in st.session_state and not st.session_state.ss_analysis_run_flag and asset_info_for_header.get('symbol'):
    st.info(f"Pronto per analizzare {asset_info_for_header['name']}. Clicca un intervallo nella tabella sopra.")
else: st.info("👋 Benvenuto! Seleziona un asset e un intervallo dalla tabella per iniziare.")

st.markdown("---")
st.caption(f"Dashboard v{APP_VERSION_FROM_CONFIG}.")
st.caption(f"Ora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}")
with st.expander("🔍 DEBUG: Stato Sessione Completo", expanded=False): 
    # ... (debug stato sessione)
    session_state_dict_for_json = {}
    for k, v_item in st.session_state.to_dict().items(): 
        if isinstance(v_item, pd.DataFrame): session_state_dict_for_json[k] = f"DataFrame shape {v_item.shape}" if v_item is not None else "None"
        elif isinstance(v_item, (datetime, date, pd.Timestamp, pd.Period)): session_state_dict_for_json[k] = str(v_item)
        else:
            try: json.dumps(v_item); session_state_dict_for_json[k] = v_item
            except (TypeError, OverflowError): session_state_dict_for_json[k] = str(v_item) 
    st.json(session_state_dict_for_json, expanded=False)
