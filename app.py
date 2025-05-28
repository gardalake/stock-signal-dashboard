# app.py - v1.6.8 (Fixes for AttributeError, NameError, rerun, API error handling - LOADED_SECRETS syntax fix)
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
CONFIG_FILE = "config.yaml"; CONFIG = {}; APP_VERSION_FROM_CONFIG = "N/A"; 
config_loaded_successfully_flag = False; yaml_error_message_for_later = None 
try:
    with open(CONFIG_FILE, 'r') as f: CONFIG = yaml.safe_load(f)
    APP_VERSION_FROM_CONFIG = CONFIG.get('version', 'v1.6.8-error-fixes (config fallback)') 
    config_loaded_successfully_flag = True
except FileNotFoundError: APP_VERSION_FROM_CONFIG = "ERRORE CONFIG - File non trovato"; print(f"CRITICAL_ERROR [app.py_module]: {CONFIG_FILE} non trovato.")
except yaml.YAMLError as e: APP_VERSION_FROM_CONFIG = "ERRORE CONFIG - YAML invalido"; yaml_error_message_for_later = e; print(f"CRITICAL_ERROR [app.py_module]: Errore parsing {CONFIG_FILE}: {e}")

# --- CONFIGURAZIONE PAGINA STREAMLIT ---
st.set_page_config(layout="wide", page_title=f"Asset Signal Dashboard {APP_VERSION_FROM_CONFIG}", page_icon="📊")
if not config_loaded_successfully_flag:
    if yaml_error_message_for_later is not None: st.error(f"ERRORE CRITICO parsing '{CONFIG_FILE}': {yaml_error_message_for_later}.")
    else: st.error(f"ERRORE CRITICO: '{CONFIG_FILE}' non trovato.")
    st.stop()
logger.info(f"{CONFIG_FILE} caricato. Versione: {APP_VERSION_FROM_CONFIG}")
if 'config_loaded_successfully' not in st.session_state: st.session_state.config_loaded_successfully = True

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
        logger.warning(f"Google AI Studio abilitato, ma URL ({google_url_secret_key}) o Token ({google_token_secret_key}) mancanti in st.secrets.")
if CONFIG.get('email_notifications', {}).get('enabled', False):
    EMAIL_SMTP_PASSWORD = st.secrets.get(email_pwd_secret_key)
    if not EMAIL_SMTP_PASSWORD:
        logger.warning(f"Email abilitate, ma password SMTP ({email_pwd_secret_key}) non in st.secrets.")

# CORREZIONE QUI: Assicurarsi che la parentesi graffa sia chiusa correttamente
LOADED_SECRETS = { 
    "ALPHA_VANTAGE_API_KEY": ALPHA_VANTAGE_API_KEY,
    google_url_secret_key: GOOGLE_AI_STUDIO_URL,
    google_token_secret_key: GOOGLE_AI_STUDIO_TOKEN,
    email_pwd_secret_key: EMAIL_SMTP_PASSWORD
} # Parentesi graffa di chiusura del dizionario

# --- DEFINIZIONI ASSET E INTERVALLI ---
TARGET_ASSETS_LIST = CONFIG.get('target_assets', [ 
    {"name": "Apple Inc.", "symbol": "AAPL", "type": "stock", "cg_id": None}, {"name": "Microsoft Corp.", "symbol": "MSFT", "type": "stock", "cg_id": None},
    {"name": "Google (Alphabet)", "symbol": "GOOGL", "type": "stock", "cg_id": None}, {"name": "Amazon.com Inc.", "symbol": "AMZN", "type": "stock", "cg_id": None},
    {"name": "NVIDIA Corp.", "symbol": "NVDA", "type": "stock", "cg_id": None}, {"name": "Tesla Inc.", "symbol": "TSLA", "type": "stock", "cg_id": None},
    {"name": "Meta Platforms", "symbol": "META", "type": "stock", "cg_id": None}, {"name": "Bitcoin", "symbol": "BTC", "type": "crypto", "cg_id": "bitcoin"},
    {"name": "Ethereum", "symbol": "ETH", "type": "crypto", "cg_id": "ethereum"}, {"name": "Solana", "symbol": "SOL", "type": "crypto", "cg_id": "solana"},
])
if not any(a['symbol'] == "NONEXISTENT_STOCK" for a in TARGET_ASSETS_LIST): TARGET_ASSETS_LIST.append({"name": "Test Stock Inesistente", "symbol": "NONEXISTENT_STOCK", "type": "stock", "cg_id": None})
if not any(a.get('cg_id') == "nonexistent_crypto_id" for a in TARGET_ASSETS_LIST if a['type'] == 'crypto'): TARGET_ASSETS_LIST.append({"name": "Test Crypto Inesistente", "symbol": "NONEXISTENT_CRYPTO", "type": "crypto", "cg_id": "nonexistent_crypto_id"})

AVAILABLE_STOCK_SYMBOLS = [asset["symbol"] for asset in TARGET_ASSETS_LIST if asset["type"] == "stock"]
AVAILABLE_CRYPTO_COINS_INFO = {asset["symbol"]: asset.get("cg_id") for asset in TARGET_ASSETS_LIST if asset["type"] == "crypto"} # Usa .get per cg_id
AVAILABLE_CRYPTO_SYMBOLS = list(AVAILABLE_CRYPTO_COINS_INFO.keys())

AV_DAILY_FUNC_NAME = CONFIG.get('alpha_vantage', {}).get('function', 'TIME_SERIES_DAILY_ADJUSTED')
AVAILABLE_INTERVALS_ACTIONS = { 
    "1H":  ("1H",   "60min", "TIME_SERIES_INTRADAY", 2,    True, 24), 
    "4H":  ("4H",   "60min", "TIME_SERIES_INTRADAY", 7,    True, 7*6), 
    "1G":  ("1D_D", "Daily", AV_DAILY_FUNC_NAME,     30,  False,30), 
    "1S":  ("1D_W", "Daily", AV_DAILY_FUNC_NAME,     7,   False,15), 
    "1M":  ("1D_M", "Daily", AV_DAILY_FUNC_NAME,     30,  False,12), 
}
DEFAULT_INTERVAL_BUTTON_LABEL = "1G" 

# --- STATO DELLA SESSIONE ---
def init_session_state_var(key, default_value):
    if key not in st.session_state:
        st.session_state[key] = default_value

first_asset_s = TARGET_ASSETS_LIST[0] if TARGET_ASSETS_LIST else {}
default_symbol_s = first_asset_s.get("symbol","N/A")
default_interval_label_s = DEFAULT_INTERVAL_BUTTON_LABEL if DEFAULT_INTERVAL_BUTTON_LABEL in AVAILABLE_INTERVALS_ACTIONS else list(AVAILABLE_INTERVALS_ACTIONS.keys())[0]
init_session_state_var('ss_current_asset_display_info', {
    "name": first_asset_s.get("name","N/A"), "symbol": default_symbol_s, "type": first_asset_s.get("type","N/A"), 
    "cg_id": first_asset_s.get("cg_id"), "interval_code": AVAILABLE_INTERVALS_ACTIONS[default_interval_label_s][0], 
    "interval_label_short": default_interval_label_s 
})
init_session_state_var('ss_asset_table_data', { asset["symbol"]: {**asset, "last_price": "N/A", "ml_signal": "N/A", "breakout_signal": "N/A"} for asset in TARGET_ASSETS_LIST })
init_session_state_var('ss_data_ohlcv_full', None); init_session_state_var('ss_data_ohlcv_display', None)
init_session_state_var('ss_features_full', None); init_session_state_var('ss_target_and_preds_full', None)
init_session_state_var('ss_final_signals_display', None); init_session_state_var('ss_trained_ml_model', None)
init_session_state_var('ss_last_signal_info_display', None); init_session_state_var('ss_analysis_run_flag', False)

# --- UI PRINCIPALE ---
st.title(f"📊 Asset Signal Dashboard"); st.caption(f"Versione: {APP_VERSION_FROM_CONFIG}"); st.markdown("---")
api_warning_placeholder_main = st.empty()
current_asset_for_display_type = st.session_state.ss_current_asset_display_info.get("type", "stock")
if current_asset_for_display_type == "stock" and not ALPHA_VANTAGE_API_KEY and CONFIG.get('alpha_vantage'): 
    api_warning_placeholder_main.warning("Chiave API Alpha Vantage non configurata.")
else: api_warning_placeholder_main.empty()

# --- TABELLA ASSET E CONTROLLI INTERVALLO ---
st.subheader("📈 Asset Overview & Analysis Triggers")
col_proportions = [0.22, 0.08, 0.12, 0.12, 0.12, 0.07, 0.07, 0.07, 0.07, 0.07] 
header_cols = st.columns(col_proportions)
headers = ["Nome", "Simbolo", "Prezzo", "Segnale ML", "Breakout", "1H", "4H", "1G", "1S", "1M"]
for col, header_text in zip(header_cols, headers): col.markdown(f"**{header_text}**")
st.markdown("<hr style='margin-top:0.5rem; margin-bottom:0.5rem;'>", unsafe_allow_html=True)

for asset_symbol_key in [a["symbol"] for a in TARGET_ASSETS_LIST]: # Itera per mantenere l'ordine originale
    asset_data_in_state = st.session_state.ss_asset_table_data.get(asset_symbol_key, {})
    asset_static_info = next((a for a in TARGET_ASSETS_LIST if a["symbol"] == asset_symbol_key), None)
    if not asset_static_info: continue

    row_cols = st.columns(col_proportions)
    row_cols[0].markdown(f"**{asset_static_info['name']}**"); row_cols[1].markdown(f"`{asset_symbol_key}`")
    row_cols[2].markdown(asset_data_in_state.get("last_price", "N/A"))
    ml_signal = asset_data_in_state.get("ml_signal", "N/A"); ml_color = "green" if ml_signal == "BUY" else "red" if ml_signal == "SELL" else ("darkorange" if ml_signal == "HOLD" else "gray")
    row_cols[3].markdown(f"<span style='color:{ml_color}; font-weight:bold;'>{ml_signal}</span>", unsafe_allow_html=True)
    breakout_signal = asset_data_in_state.get("breakout_signal", "N/A"); breakout_color = "blue" if breakout_signal == "BULLISH" else ("orange" if breakout_signal == "BEARISH" else "gray")
    row_cols[4].markdown(f"<span style='color:{breakout_color};'>{breakout_signal}</span>", unsafe_allow_html=True)
    
    interval_button_labels_short_ordered = list(AVAILABLE_INTERVALS_ACTIONS.keys()) 
    for i, short_label in enumerate(interval_button_labels_short_ordered):
        if 5 + i < len(row_cols): # Assicura che ci sia una colonna
            interval_details_tuple_btn = AVAILABLE_INTERVALS_ACTIONS[short_label]
            interval_code_for_button = interval_details_tuple_btn[0]
            button_key = f"btn_{asset_symbol_key}_{short_label}"
            
            is_selected_button = (st.session_state.ss_current_asset_display_info["symbol"] == asset_symbol_key and
                                  st.session_state.ss_current_asset_display_info["interval_label_short"] == short_label)
            button_type = "primary" if is_selected_button else "secondary"

            if row_cols[5+i].button(short_label, key=button_key, use_container_width=True, type=button_type):
                logger.info(f"Bottone cliccato: Asset {asset_symbol_key}, Intervallo Label: {short_label}, Codice: {interval_code_for_button}")
                st.session_state.ss_current_asset_display_info = {
                    "name": asset_static_info["name"], "symbol": asset_symbol_key,
                    "type": asset_static_info["type"], "cg_id": asset_static_info.get("cg_id"),
                    "interval_code": interval_code_for_button, "interval_label_short": short_label 
                }
                st.session_state.ss_analysis_run_flag = True 
                st.session_state.ss_data_ohlcv_full = None; st.session_state.ss_data_ohlcv_display = None; st.session_state.ss_features_full = None; st.session_state.ss_target_and_preds_full = None; st.session_state.ss_final_signals_display = None; st.session_state.ss_trained_ml_model = None; st.session_state.ss_data_ohlcv_daily_for_ml = None 
                st.rerun() 
    st.markdown("<hr style='margin-top:0.2rem; margin-bottom:0.2rem;'>", unsafe_allow_html=True)
st.markdown("---") 

# --- PIPELINE DI ELABORAZIONE ---
if st.session_state.get('ss_analysis_run_flag', False):
    current_asset_info_pipeline = st.session_state.ss_current_asset_display_info
    current_interval_label_short_pipeline = current_asset_info_pipeline["interval_label_short"] 
    interval_details_pipeline = AVAILABLE_INTERVALS_ACTIONS.get(current_interval_label_short_pipeline)
    if not interval_details_pipeline:
        logger.error(f"Dettagli intervallo non trovati per {current_interval_label_short_pipeline}, uso default 1G.")
        interval_details_pipeline = AVAILABLE_INTERVALS_ACTIONS[DEFAULT_INTERVAL_BUTTON_LABEL]
    
    interval_code_p, av_api_interval_p, av_api_function_p, cg_api_days_granularity_p, interval_is_intraday_p, display_units_ago_p = interval_details_pipeline

    _display_end_date_dt_p = datetime.now() 
    if interval_code_p == "1H": _display_start_date_dt_p = _display_end_date_dt_p - timedelta(hours=display_units_ago_p)
    elif interval_code_p == "4H": _display_start_date_dt_p = _display_end_date_dt_p - timedelta(hours=display_units_ago_p * 4) 
    elif interval_code_p.startswith("1D"): 
        time_unit_char = interval_code_p.split('_')[-1][-1]
        num_part_str = interval_code_p.split('_')[-1][:-1] if len(interval_code_p.split('_')[-1]) > 1 else str(display_units_ago_p)
        num_val = int(num_part_str) if num_part_str.isdigit() else display_units_ago_p
        if time_unit_char == "W": _display_start_date_dt_p = _display_end_date_dt_p - timedelta(weeks=num_val)
        elif time_unit_char == "M": _display_start_date_dt_p = _display_end_date_dt_p - timedelta(days=num_val * 30) 
        elif time_unit_char == "Y": _display_start_date_dt_p = _display_end_date_dt_p - timedelta(days=num_val * 365)
        else: _display_start_date_dt_p = _display_end_date_dt_p - timedelta(days=num_val) 
    else: _display_start_date_dt_p = _display_end_date_dt_p - timedelta(days=30) 

    MIN_DAYS_FOR_ML_AND_TA_p = CONFIG.get('ml_model', {}).get('min_days_for_indicators_and_training', 200)
    days_for_ml_history_p = MIN_DAYS_FOR_ML_AND_TA_p if not interval_is_intraday_p else 30 
    _api_data_load_start_date_dt_p = _display_start_date_dt_p.date() - timedelta(days=days_for_ml_history_p)
    _av_outputsize_param_p = "compact" if interval_is_intraday_p and current_asset_info_pipeline["type"] == "stock" else "full"
    _cg_days_to_fetch_param_p = (date.today() - _api_data_load_start_date_dt_p).days + 1
    if _cg_days_to_fetch_param_p <= 0: _cg_days_to_fetch_param_p = days_for_ml_history_p 
    if interval_is_intraday_p and current_asset_info_pipeline["type"] == "crypto":
        _cg_days_to_fetch_param_p = cg_api_days_granularity_p 
    
    log_container = st.container()
    with log_container:
        st.markdown(f"### ⚙️ Analisi per: {current_asset_info_pipeline['name']} ({current_asset_info_pipeline['symbol']}) - Visualizzazione: {current_asset_info_pipeline['interval_label_short']}")
        progress_bar = st.progress(0, text="Inizio...")
        logger.info(f"Caricamento dati. Display: {_display_start_date_dt_p.strftime('%Y-%m-%d %H:%M')} a {_display_end_date_dt_p.strftime('%Y-%m-%d %H:%M')}. API load start: {_api_data_load_start_date_dt_p.strftime('%Y-%m-%d')}")
        
        asset_to_fetch = current_asset_info_pipeline
        progress_bar.progress(5, text=f"Caricamento dati display ({current_asset_info_pipeline['interval_label_short']})...")
        if asset_to_fetch["type"] == "stock":
            if not ALPHA_VANTAGE_API_KEY: st.error("Chiave API AV mancante."); st.session_state.ss_data_ohlcv_display = None
            else:
                av_call_params_display = {}
                if av_api_function_p == "TIME_SERIES_INTRADAY": av_call_params_display['av_interval'] = av_api_interval_p
                outputsize_display = "compact" if interval_is_intraday_p else "full" # av_outputsize_param_p era per storico ML
                st.session_state.ss_data_ohlcv_display = get_stock_data(ALPHA_VANTAGE_API_KEY, asset_to_fetch["symbol"], av_api_function_p, outputsize_display, **av_call_params_display)
        elif asset_to_fetch["type"] == "crypto":
            # Per il display, usiamo cg_api_days_granularity_p se intraday, altrimenti calcoliamo i giorni per il display.
            days_for_cg_display = cg_api_days_granularity_p if interval_is_intraday_p else (date.today() - _display_start_date_dt_p.date()).days + 1
            if days_for_cg_display <=0 : days_for_cg_display = 30 # Fallback per display
            logger.debug(f"Caricamento crypto display - Giorni: {days_for_cg_display}, intervallo: {interval_code_p}")
            st.session_state.ss_data_ohlcv_display = get_crypto_data(asset_to_fetch.get("cg_id"), CONFIG.get('coingecko',{}).get('vs_currency', 'usd'), days_for_cg_display, interval_code_p)

        if st.session_state.ss_data_ohlcv_display is not None and not st.session_state.ss_data_ohlcv_display.empty:
            _start_dt_disp_filt_pd = pd.to_datetime(_display_start_date_dt_p); _end_dt_disp_filt_pd = pd.to_datetime(_display_end_date_dt_p)
            if not isinstance(st.session_state.ss_data_ohlcv_display.index, pd.DatetimeIndex): st.session_state.ss_data_ohlcv_display.index = pd.to_datetime(st.session_state.ss_data_ohlcv_display.index)
            df_to_filt_disp = st.session_state.ss_data_ohlcv_display
            # Filtra di nuovo i dati display per assicurare che siano esattamente nell'intervallo,
            # dato che le API potrebbero restituire più dati del richiesto (es. "compact" per AV intraday)
            if interval_is_intraday_p: st.session_state.ss_data_ohlcv_display = df_to_filt_disp[(df_to_filt_disp.index >= _start_dt_disp_filt_pd) & (df_to_filt_disp.index < _end_dt_disp_filt_pd + pd.Timedelta(days=1))].copy()
            else: st.session_state.ss_data_ohlcv_display = df_to_filt_disp[(df_to_filt_disp.index.normalize() >= _start_dt_disp_filt_pd.normalize()) & (df_to_filt_disp.index.normalize() <= _end_dt_disp_filt_pd.normalize())].copy()
            if st.session_state.ss_data_ohlcv_display.empty: st.warning(f"Nessun dato per display dopo filtraggio.")
            else: st.success(f"Dati per display pronti. Shape: {st.session_state.ss_data_ohlcv_display.shape}")
        else: st.error(f"Fallimento caricamento dati display per {asset_to_fetch['symbol']}.")

        progress_bar.progress(15, text=f"Caricamento dati giornalieri per ML ({asset_to_fetch['symbol']})...")
        if asset_to_fetch["type"] == "stock":
            if not ALPHA_VANTAGE_API_KEY: st.error("Chiave API AV mancante per dati ML."); st.session_state.ss_data_ohlcv_daily_for_ml = None
            else: st.session_state.ss_data_ohlcv_daily_for_ml = get_stock_data(ALPHA_VANTAGE_API_KEY, asset_to_fetch["symbol"], AV_DAILY_FUNC_NAME, "full")
        elif asset_to_fetch["type"] == "crypto":
            st.session_state.ss_data_ohlcv_daily_for_ml = get_crypto_data(asset_to_fetch.get("cg_id"), CONFIG.get('coingecko',{}).get('vs_currency', 'usd'), _cg_days_to_fetch_ml_p, "1D_D") 

        if st.session_state.ss_data_ohlcv_daily_for_ml is not None and not st.session_state.ss_data_ohlcv_daily_for_ml.empty:
            logger.info(f"Dati giornalieri per ML caricati. Shape: {st.session_state.ss_data_ohlcv_daily_for_ml.shape}")
            _ml_end_dt_filt = pd.to_datetime(date.today()) # Usa data odierna per fine periodo ML
            _ml_start_dt_filt = pd.to_datetime(_ml_data_load_start_date_dt_p)
            if not isinstance(st.session_state.ss_data_ohlcv_daily_for_ml.index, pd.DatetimeIndex): st.session_state.ss_data_ohlcv_daily_for_ml.index = pd.to_datetime(st.session_state.ss_data_ohlcv_daily_for_ml.index)
            st.session_state.ss_data_ohlcv_daily_for_ml = st.session_state.ss_data_ohlcv_daily_for_ml[
                (st.session_state.ss_data_ohlcv_daily_for_ml.index.normalize() >= _ml_start_dt_filt.normalize()) &
                (st.session_state.ss_data_ohlcv_daily_for_ml.index.normalize() <= _ml_end_dt_filt.normalize())
            ].copy()
            if st.session_state.ss_data_ohlcv_daily_for_ml.empty: st.warning("Dati giornalieri ML vuoti dopo filtro date.")
        else:
            st.error(f"Fallimento caricamento dati giornalieri per ML ({asset_to_fetch['symbol']}). Pipeline ML non può procedere.")
            st.session_state.ss_analysis_run_flag = False; progress_bar.empty(); st.stop()

        if st.session_state.ss_data_ohlcv_daily_for_ml is not None and not st.session_state.ss_data_ohlcv_daily_for_ml.empty:
            df_ml_input = st.session_state.ss_data_ohlcv_daily_for_ml
            progress_bar.progress(25, text="Calcolo feature ML (su dati giornalieri)...")
            features_for_ml = calculate_technical_features(df_ml_input)
            st.session_state.ss_features_full = features_for_ml # Salva feature giornaliere per possibile uso nel grafico display

            if features_for_ml.empty or len(features_for_ml) < MIN_DAYS_FOR_ML_AND_TA_p / 4 : # Soglia più flessibile
                st.error("Fallimento calcolo feature ML o dati insuff.")
            else:
                pred_horizon_days_ml = CONFIG.get('ml_model', {}).get('prediction_target_horizon_days', 3)
                df_target_ml = create_prediction_targets(features_for_ml, horizon=pred_horizon_days_ml)
                target_col_name_ml = f'target_{pred_horizon_days_ml}d_pct_change'
                feature_cols_config_ml = CONFIG.get('ml_model',{}).get('feature_columns_for_training', ['MA20', 'MA50', 'RSI'])
                feature_cols_actual_ml = [col for col in feature_cols_config_ml if col in df_target_ml.columns]

                if feature_cols_actual_ml and target_col_name_ml in df_target_ml.columns:
                    st.session_state.ss_trained_ml_model = train_random_forest_model(df_target_ml, feature_cols_actual_ml, target_col_name_ml)
                    if st.session_state.ss_trained_ml_model:
                        preds_series_ml = generate_model_predictions(st.session_state.ss_trained_ml_model, df_target_ml, feature_cols_actual_ml)
                        if preds_series_ml is not None:
                            st.session_state.ss_target_and_preds_full = df_target_ml.copy() # Giornaliero
                            pred_col_name_ml = f'prediction_{pred_horizon_days_ml}d_pct_change'
                            st.session_state.ss_target_and_preds_full[pred_col_name_ml] = preds_series_ml
                            
                            ml_signals_df_full = generate_signals_from_ml_predictions(st.session_state.ss_target_and_preds_full, pred_col_name_ml, CONFIG.get('signal_logic',{}).get('buy_threshold_change', 0.005), CONFIG.get('signal_logic',{}).get('sell_threshold_change', -0.005))
                            breakout_df_full = detect_breakout_signals(features_for_ml) # features_for_ml è giornaliero
                            signals_combined_daily = combine_signals(ml_signals_df_full, breakout_df_full)
                            signals_combined_daily = apply_trading_spreads(signals_combined_daily, asset_to_fetch["type"], CONFIG.get('spreads',{}))

                            if not signals_combined_daily.empty:
                                last_daily_signal_row = signals_combined_daily.iloc[-1]
                                asset_sym_update = asset_to_fetch["symbol"]
                                if asset_sym_update in st.session_state.ss_asset_table_data:
                                    st.session_state.ss_asset_table_data[asset_sym_update]["ml_signal"] = last_daily_signal_row.get('ml_signal', 'N/A')
                                    st.session_state.ss_asset_table_data[asset_sym_update]["breakout_signal"] = last_daily_signal_row.get('breakout_signal', 'N/A')
                                    st.session_state.ss_asset_table_data[asset_sym_update]["last_price"] = f"{last_daily_signal_row.get('Close', 0.0):.2f}" if 'Close' in last_daily_signal_row else "N/A"
                            
                            if st.session_state.ss_data_ohlcv_display is not None and not st.session_state.ss_data_ohlcv_display.empty:
                                signals_to_merge = signals_combined_daily[['ml_signal', 'breakout_signal']].copy()
                                signals_to_merge.index = signals_to_merge.index.normalize()
                                df_display_with_date_idx = st.session_state.ss_data_ohlcv_display.copy()
                                df_display_with_date_idx['temp_date_idx_for_signal_merge'] = df_display_with_date_idx.index.normalize()
                                merged_for_display = pd.merge(df_display_with_date_idx, signals_to_merge, left_on='temp_date_idx_for_signal_merge', right_index=True, how='left')
                                merged_for_display.drop(columns=['temp_date_idx_for_signal_merge'], inplace=True, errors='ignore')
                                merged_for_display[['ml_signal', 'breakout_signal']] = merged_for_display[['ml_signal', 'breakout_signal']].fillna(method='ffill')
                                st.session_state.ss_final_signals_display = merged_for_display.copy()
                                st.success(f"Segnali allineati ai dati display. Shape: {st.session_state.ss_final_signals_display.shape}")
                                if not st.session_state.ss_final_signals_display.empty:
                                    last_disp_sig = st.session_state.ss_final_signals_display.iloc[-1]
                                    # L'ultimo segnale *mostrato* è quello della candela display, ma il segnale *calcolato* è quello giornaliero
                                    st.session_state.ss_last_signal_info_display = {"ticker": asset_to_fetch["symbol"], 
                                                                                "date": last_disp_sig.name.strftime('%Y-%m-%d %H:%M:%S') if isinstance(last_disp_sig.name, pd.Timestamp) else str(last_disp_sig.name), 
                                                                                "ml_signal": last_daily_signal_row.get('ml_signal'), 
                                                                                "breakout_signal": last_daily_signal_row.get('breakout_signal'), 
                                                                                "close_price": f"{last_disp_sig.get('Close',0):.2f}"}
                                    if last_daily_signal_row.get('ml_signal') == 'BUY': play_buy_signal_sound(CONFIG.get('sound_utils',{}))
                                    elif last_daily_signal_row.get('ml_signal') == 'SELL': play_sell_signal_sound(CONFIG.get('sound_utils',{}))
        # --- FINE BLOCCO PIPELINE ---
        progress_bar.progress(100, text="Analisi completata!")
        time.sleep(0.5); progress_bar.empty() 
    if st.session_state.get('ss_analysis_run_flag', False): st.session_state.ss_analysis_run_flag = False

# --- AREA PRINCIPALE PER VISUALIZZAZIONE RISULTATI ---
st.markdown("---")
asset_info_for_header = st.session_state.ss_current_asset_display_info
st.header(f"📊 Risultati per: {asset_info_for_header.get('name')} ({asset_info_for_header.get('symbol')}) - Visualizzazione: {asset_info_for_header.get('interval_label_short')}")
if st.session_state.get('ss_final_signals_display') is not None and isinstance(st.session_state.ss_final_signals_display, pd.DataFrame) and not st.session_state.ss_final_signals_display.empty:
    if st.session_state.ss_last_signal_info_display:
        st.subheader("📢 Ultimo Segnale Calcolato (Base Giornaliera):")
        sig_info = st.session_state.ss_last_signal_info_display; ml_color = "green" if sig_info['ml_signal'] == "BUY" else "red" if sig_info['ml_signal'] == "SELL" else "gray"; breakout_color = "blue" if sig_info['breakout_signal'] == "BULLISH" else "orange" if sig_info['breakout_signal'] == "BEARISH" else "gray"
        st.markdown(f"*   **Data Segnale (Giornaliero):** `{sig_info['date'].split(' ')[0]}` *   **Segnale ML:** <span style='color:{ml_color};'>{sig_info['ml_signal']}</span> *   **Breakout:** <span style='color:{breakout_color};'>{sig_info['breakout_signal']}</span> *   **Prezzo Chiusura (Candela Display):** `{sig_info['close_price']}`", unsafe_allow_html=True)
    st.markdown("---"); st.subheader("📈 Grafico Interattivo")
    df_features_for_chart = pd.DataFrame() 
    # Per il grafico, vogliamo OHLCV dalla granularità di display (ss_data_ohlcv_display)
    # e le MA calcolate sulla stessa granularità (se possibile, o giornaliere se è l'unica opzione)
    # Attualmente ss_features_full contiene feature GIORNALIERE.
    # Se il display è intraday, le MA giornaliere potrebbero non essere l'ideale da plottare direttamente.
    # Per ora, uniamo le MA giornaliere se l'indice combacia, altrimenti il grafico non avrà MA.
    df_chart_input_ohlcv = st.session_state.ss_data_ohlcv_display.copy() if st.session_state.ss_data_ohlcv_display is not None else pd.DataFrame()

    if not df_chart_input_ohlcv.empty and st.session_state.ss_features_full is not None and not st.session_state.ss_features_full.empty:
        ma_cols_to_add = CONFIG.get('visualization',{}).get('show_ma_periods', []) # Lista di stringhe MA, es. ['MA20', 'MA50']
        # Filtra solo le colonne MA che esistono in ss_features_full
        actual_ma_cols = [col for col in ma_cols_to_add if col in st.session_state.ss_features_full.columns]
        if actual_ma_cols:
            # Unisci le MA giornaliere ai dati di display
            df_ma_to_merge = st.session_state.ss_features_full[actual_ma_cols].copy()
            df_ma_to_merge.index = df_ma_to_merge.index.normalize() # Assicura indice giornaliero
            
            temp_idx_name = "temp_date_for_ma_merge"
            df_chart_input_ohlcv[temp_idx_name] = df_chart_input_ohlcv.index.normalize()
            df_chart_input_ohlcv = pd.merge(df_chart_input_ohlcv, df_ma_to_merge, left_on=temp_idx_name, right_index=True, how='left')
            df_chart_input_ohlcv.drop(columns=[temp_idx_name], inplace=True, errors='ignore')

    if not df_chart_input_ohlcv.empty:
        chart_fig = create_main_stock_chart(df_chart_input_ohlcv, st.session_state.ss_final_signals_display, asset_info_for_header["symbol"], CONFIG.get('visualization',{}).get('show_ma_periods', []))
        st.plotly_chart(chart_fig, use_container_width=True)
    else: 
        if st.session_state.get('ss_data_ohlcv_full') is not None: st.warning("Dati grafico insuff.")
    with st.expander("👁️ Dati Tabellari (ultimi 100 record dell'intervallo)"):
        if st.session_state.ss_data_ohlcv_display is not None and not st.session_state.ss_data_ohlcv_display.empty: st.dataframe(st.session_state.ss_data_ohlcv_display.tail(100))
elif 'ss_analysis_run_flag' in st.session_state and not st.session_state.ss_analysis_run_flag and asset_info_for_header.get('symbol'):
    st.info(f"Pronto per analizzare {asset_info_for_header['name']}. Clicca un intervallo nella tabella.")
else: st.info("👋 Benvenuto! Seleziona un asset e un intervallo dalla tabella.")
st.markdown("---")
st.caption(f"Dashboard v{APP_VERSION_FROM_CONFIG}.")
st.caption(f"Ora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}")
with st.expander("🔍 DEBUG: Stato Sessione Completo", expanded=False): 
    session_state_dict_for_json = {}
    for k, v_item in st.session_state.to_dict().items(): 
        if isinstance(v_item, pd.DataFrame): session_state_dict_for_json[k] = f"DataFrame shape {v_item.shape if v_item is not None else 'None'}"
        elif isinstance(v_item, (datetime, date, pd.Timestamp, pd.Period)): session_state_dict_for_json[k] = str(v_item)
        else:
            try: json.dumps(v_item); session_state_dict_for_json[k] = v_item
            except (TypeError, OverflowError): session_state_dict_for_json[k] = str(v_item) 
    st.json(session_state_dict_for_json, expanded=False)
