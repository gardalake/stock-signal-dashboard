# app.py - v1.6.7 (UI Refactor - Fix experimental_rerun to rerun)
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

if not config_loaded_successfully_flag:
    if yaml_error_message_for_later is not None:
        st.error(f"ERRORE CRITICO nel parsing di '{CONFIG_FILE}': {yaml_error_message_for_later}.")
        st.stop()
    else: 
        st.error(f"ERRORE CRITICO: '{CONFIG_FILE}' non trovato.")
        st.stop()
logger.info(f"{CONFIG_FILE} caricato. Versione da config: {APP_VERSION_FROM_CONFIG}")
if 'config_loaded_successfully' not in st.session_state: 
    st.session_state.config_loaded_successfully = True


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
        logger.warning(f"Google AI Studio abilitato, ma URL ({google_url_secret_key}) o Token ({google_token_secret_key}) mancanti.")
if CONFIG.get('email_notifications', {}).get('enabled', False):
    EMAIL_SMTP_PASSWORD = st.secrets.get(email_pwd_secret_key)
    if not EMAIL_SMTP_PASSWORD:
        logger.warning(f"Email abilitate, ma password SMTP ({email_pwd_secret_key}) non in st.secrets.")
LOADED_SECRETS = { 
    "ALPHA_VANTAGE_API_KEY": ALPHA_VANTAGE_API_KEY,
    google_url_secret_key: GOOGLE_AI_STUDIO_URL,
    google_token_secret_key: GOOGLE_AI_STUDIO_TOKEN,
    email_pwd_secret_key: EMAIL_SMTP_PASSWORD
}

# --- DEFINIZIONI ASSET E INTERVALLI ---
TARGET_ASSETS_LIST = CONFIG.get('target_assets', [ 
    {"name": "Apple Inc.", "symbol": "AAPL", "type": "stock", "cg_id": None},
    {"name": "Microsoft Corp.", "symbol": "MSFT", "type": "stock", "cg_id": None},
    {"name": "Google (Alphabet)", "symbol": "GOOGL", "type": "stock", "cg_id": None},
    {"name": "Amazon.com Inc.", "symbol": "AMZN", "type": "stock", "cg_id": None},
    {"name": "NVIDIA Corp.", "symbol": "NVDA", "type": "stock", "cg_id": None},
    {"name": "Tesla Inc.", "symbol": "TSLA", "type": "stock", "cg_id": None},
    {"name": "Meta Platforms", "symbol": "META", "type": "stock", "cg_id": None},
    {"name": "Bitcoin", "symbol": "BTC", "type": "crypto", "cg_id": "bitcoin"},
    {"name": "Ethereum", "symbol": "ETH", "type": "crypto", "cg_id": "ethereum"},
    {"name": "Solana", "symbol": "SOL", "type": "crypto", "cg_id": "solana"},
])
if not any(a['symbol'] == "NONEXISTENT_STOCK" for a in TARGET_ASSETS_LIST):
    TARGET_ASSETS_LIST.append({"name": "Test Stock Inesistente", "symbol": "NONEXISTENT_STOCK", "type": "stock", "cg_id": None})
if not any(a.get('cg_id') == "nonexistent_crypto_id" for a in TARGET_ASSETS_LIST if a['type'] == 'crypto'):
    TARGET_ASSETS_LIST.append({"name": "Test Crypto Inesistente", "symbol": "NONEXISTENT_CRYPTO", "type": "crypto", "cg_id": "nonexistent_crypto_id"})

AVAILABLE_STOCK_SYMBOLS = [asset["symbol"] for asset in TARGET_ASSETS_LIST if asset["type"] == "stock"]
AVAILABLE_CRYPTO_COINS_INFO = {asset["symbol"]: asset["cg_id"] for asset in TARGET_ASSETS_LIST if asset["type"] == "crypto" and asset.get("cg_id")}
AVAILABLE_CRYPTO_SYMBOLS = list(AVAILABLE_CRYPTO_COINS_INFO.keys())

AVAILABLE_INTERVALS_ACTIONS = {
    "1H":  ("1H",   "60min", "TIME_SERIES_INTRADAY", 2,    True, 24), 
    "4H":  ("4H",   "60min", "TIME_SERIES_INTRADAY", 7,    True, 7*6), 
    "1G":  ("1D_1D", "Daily", CONFIG.get('alpha_vantage', {}).get('function', 'TIME_SERIES_DAILY_ADJUSTED'), 30,  False,30), # Modificato codice per "1 Giorno"
    "1S":  ("1D_1W","Daily", CONFIG.get('alpha_vantage', {}).get('function', 'TIME_SERIES_DAILY_ADJUSTED'), 7,   False,7*15), # Mostra 15 settimane
    "1M":  ("1D_1M","Daily", CONFIG.get('alpha_vantage', {}).get('function', 'TIME_SERIES_DAILY_ADJUSTED'), 30,  False,12), # Mostra 12 mesi
}
DEFAULT_INTERVAL_BUTTON_LABEL = "1G" 

# --- STATO DELLA SESSIONE ---
if 'ss_current_asset_display_info' not in st.session_state: 
    first_asset = TARGET_ASSETS_LIST[0] if TARGET_ASSETS_LIST else {}
    default_symbol = first_asset.get("symbol","N/A")
    # Assicura che il default interval label sia una chiave valida
    default_interval_label_for_state = DEFAULT_INTERVAL_BUTTON_LABEL if DEFAULT_INTERVAL_BUTTON_LABEL in AVAILABLE_INTERVALS_ACTIONS else list(AVAILABLE_INTERVALS_ACTIONS.keys())[0]
    
    st.session_state.ss_current_asset_display_info = {
        "name": first_asset.get("name","N/A"), "symbol": default_symbol,
        "type": first_asset.get("type","N/A"), "cg_id": first_asset.get("cg_id"),
        "interval_code": AVAILABLE_INTERVALS_ACTIONS[default_interval_label_for_state][0], 
        "interval_label_short": default_interval_label_for_state 
    }
if 'ss_asset_table_data' not in st.session_state:
    st.session_state.ss_asset_table_data = {
        asset["symbol"]: {**asset, "last_price": "N/A", "ml_signal": "N/A", "breakout_signal": "N/A"}
        for asset in TARGET_ASSETS_LIST
    }
for key in ['ss_data_ohlcv_full', 'ss_data_ohlcv_display', 'ss_features_full', 
            'ss_target_and_preds_full', 'ss_final_signals_display', 
            'ss_trained_ml_model', 'ss_last_signal_info_display', 'ss_analysis_run_flag']:
    if key not in st.session_state:
        st.session_state[key] = None if "df" in key or "model" in key else False

# --- UI PRINCIPALE ---
st.title(f"📊 Asset Signal Dashboard")
st.caption(f"Versione: {APP_VERSION_FROM_CONFIG}")
st.markdown("---")

api_warning_placeholder_main = st.empty()
# Determina se mostrare il warning per la chiave AV in base all'asset attualmente selezionato per il display
current_asset_for_display_type = st.session_state.ss_current_asset_display_info.get("type", "stock")
if current_asset_for_display_type == "stock" and not ALPHA_VANTAGE_API_KEY and CONFIG.get('alpha_vantage'): 
    api_warning_placeholder_main.warning("Chiave API Alpha Vantage non configurata. Dati per azioni non disponibili.")
else:
    api_warning_placeholder_main.empty()

# --- TABELLA ASSET E CONTROLLI INTERVALLO ---
st.subheader("📈 Asset Overview & Analysis Triggers")
col_proportions = [0.22, 0.08, 0.12, 0.12, 0.12, 0.07, 0.07, 0.07, 0.07, 0.07] 
header_cols = st.columns(col_proportions)
headers = ["Nome", "Simbolo", "Prezzo", "Segnale ML", "Breakout", "1H", "4H", "1G", "1S", "1M"]
for col, header_text in zip(header_cols, headers):
    col.markdown(f"**{header_text}**")
st.markdown("<hr style='margin-top:0.5rem; margin-bottom:0.5rem;'>", unsafe_allow_html=True)

for asset_symbol_key in [a["symbol"] for a in TARGET_ASSETS_LIST]:
    asset_data_in_state = st.session_state.ss_asset_table_data.get(asset_symbol_key, {})
    asset_static_info = next((a for a in TARGET_ASSETS_LIST if a["symbol"] == asset_symbol_key), None)
    if not asset_static_info: continue

    row_cols = st.columns(col_proportions)
    row_cols[0].markdown(f"**{asset_static_info['name']}**")
    row_cols[1].markdown(f"`{asset_symbol_key}`")
    row_cols[2].markdown(asset_data_in_state.get("last_price", "N/A"))
    
    ml_signal = asset_data_in_state.get("ml_signal", "N/A")
    ml_color = "green" if ml_signal == "BUY" else "red" if ml_signal == "SELL" else ("darkorange" if ml_signal == "HOLD" else "gray")
    row_cols[3].markdown(f"<span style='color:{ml_color}; font-weight:bold;'>{ml_signal}</span>", unsafe_allow_html=True)

    breakout_signal = asset_data_in_state.get("breakout_signal", "N/A")
    breakout_color = "blue" if breakout_signal == "BULLISH" else ("orange" if breakout_signal == "BEARISH" else "gray")
    row_cols[4].markdown(f"<span style='color:{breakout_color};'>{breakout_signal}</span>", unsafe_allow_html=True)
    
    interval_button_labels_short_ordered = list(AVAILABLE_INTERVALS_ACTIONS.keys()) 
    for i, short_label in enumerate(interval_button_labels_short_ordered):
        # Assicurati che ci sia una colonna disponibile
        if 5 + i < len(row_cols):
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
                st.session_state.ss_data_ohlcv_full = None; st.session_state.ss_data_ohlcv_display = None
                st.session_state.ss_features_full = None; st.session_state.ss_target_and_preds_full = None
                st.session_state.ss_final_signals_display = None; st.session_state.ss_trained_ml_model = None
                st.rerun() # CORRETTO DA experimental_rerun()
    st.markdown("<hr style='margin-top:0.2rem; margin-bottom:0.2rem;'>", unsafe_allow_html=True)
st.markdown("---") 

# --- LOGICA CALCOLO DATE E PARAMETRI API ---
current_asset_info_pipeline = st.session_state.ss_current_asset_display_info
current_interval_label_short_pipeline = current_asset_info_pipeline["interval_label_short"] # Usa l'etichetta breve del bottone
interval_details_pipeline = AVAILABLE_INTERVALS_ACTIONS.get(current_interval_label_short_pipeline)

if not interval_details_pipeline:
    logger.error(f"Dettagli intervallo non trovati per etichetta breve: {current_interval_label_short_pipeline}. Uso default 1G.")
    interval_details_pipeline = AVAILABLE_INTERVALS_ACTIONS[DEFAULT_INTERVAL_BUTTON_LABEL]

interval_code_p, av_api_interval_p, av_api_function_p, cg_api_days_granularity_p, interval_is_intraday_p, display_units_ago_p = interval_details_pipeline

_display_end_date_dt_p = datetime.now() 
_display_start_date_dt_p = None
# Calcolo date di display basato su display_units_ago_p
if interval_code_p == "1H": _display_start_date_dt_p = _display_end_date_dt_p - timedelta(hours=display_units_ago_p)
elif interval_code_p == "4H": _display_start_date_dt_p = _display_end_date_dt_p - timedelta(hours=display_units_ago_p * 4) # display_units_ago_p è numero di "blocchi da 4h"
elif interval_code_p.startswith("1D"): 
    _display_start_date_dt_p = _display_end_date_dt_p - timedelta(days=display_units_ago_p)
else: _display_start_date_dt_p = _display_end_date_dt_p - timedelta(days=30) 

MIN_DAYS_FOR_ML_AND_TA_p = CONFIG.get('ml_model', {}).get('min_days_for_indicators_and_training', 200)
# Per intraday, carica uno storico più breve per ML (es. 30 giorni di dati orari)
days_for_ml_history = MIN_DAYS_FOR_ML_AND_TA_p if not interval_is_intraday_p else 30 
_api_data_load_start_date_dt_p = _display_start_date_dt_p.date() - timedelta(days=days_for_ml_history)

_av_outputsize_param_p = "compact" if interval_is_intraday_p and current_asset_info_pipeline["type"] == "stock" else "full"
_cg_days_to_fetch_param_p = (date.today() - _api_data_load_start_date_dt_p).days + 1
if _cg_days_to_fetch_param_p <= 0: _cg_days_to_fetch_param_p = days_for_ml_history 
if interval_is_intraday_p and current_asset_info_pipeline["type"] == "crypto":
    _cg_days_to_fetch_param_p = cg_api_days_granularity_p 

# --- PIPELINE DI ELABORAZIONE ---
if st.session_state.get('ss_analysis_run_flag', False):
    log_container = st.container()
    with log_container:
        # ... (Pipeline di analisi come prima, assicurati che sia tutta indentata correttamente sotto questo 'with')
        st.markdown(f"### ⚙️ Analisi per: {current_asset_info_pipeline['name']} ({current_asset_info_pipeline['symbol']}) - {current_asset_info_pipeline['interval_label_short']}")
        progress_bar = st.progress(0, text="Inizio...") # ...
        # (Ometto il resto della pipeline per brevità, è identica a quella precedente)
        # --- INIZIO BLOCCO PIPELINE (ASSICURATI CHE L'INDENTAZIONE SIA CORRETTA) ---
        logger.info(f"Caricamento dati. Display: {_display_start_date_dt_p.strftime('%Y-%m-%d %H:%M')} a {_display_end_date_dt_p.strftime('%Y-%m-%d %H:%M')}. API load start: {_api_data_load_start_date_dt_p.strftime('%Y-%m-%d')}")
        progress_bar.progress(10, text=f"Caricamento storico per {current_asset_info_pipeline['symbol']}...")
        if current_asset_info_pipeline["type"] == "stock":
            if not ALPHA_VANTAGE_API_KEY: st.error("Chiave API AV mancante."); st.session_state.ss_data_ohlcv_full = None
            else:
                av_call_params = {}
                if av_api_function_p == "TIME_SERIES_INTRADAY": av_call_params['av_interval'] = av_api_interval_p
                st.session_state.ss_data_ohlcv_full = get_stock_data(ALPHA_VANTAGE_API_KEY, current_asset_info_pipeline["symbol"], av_api_function_p, _av_outputsize_param_p, **av_call_params)
        elif current_asset_info_pipeline["type"] == "crypto":
            cg_id_to_fetch = current_asset_info_pipeline.get("cg_id")
            if not cg_id_to_fetch: 
                st.error(f"ID CoinGecko non trovato per {current_asset_info_pipeline['symbol']}"); st.session_state.ss_data_ohlcv_full = None
            else:
                st.session_state.ss_data_ohlcv_full = get_crypto_data(cg_id_to_fetch, CONFIG.get('coingecko',{}).get('vs_currency', 'usd'), _cg_days_to_fetch_param_p, current_interval_code_pipeline)
        
        if st.session_state.ss_data_ohlcv_full is not None and not st.session_state.ss_data_ohlcv_full.empty:
            logger.info(f"Storico completo caricato. Shape: {st.session_state.ss_data_ohlcv_full.shape}")
            _start_dt_display_filter_pd = pd.to_datetime(_display_start_date_dt_p); _end_dt_display_filter_pd = pd.to_datetime(_display_end_date_dt_p)
            if not isinstance(st.session_state.ss_data_ohlcv_full.index, pd.DatetimeIndex): st.session_state.ss_data_ohlcv_full.index = pd.to_datetime(st.session_state.ss_data_ohlcv_full.index)
            df_to_filter = st.session_state.ss_data_ohlcv_full
            if interval_is_intraday_p: st.session_state.ss_data_ohlcv_display = df_to_filter[(df_to_filter.index >= _start_dt_display_filter_pd) & (df_to_filter.index < _end_dt_display_filter_pd + pd.Timedelta(days=1))].copy()
            else: st.session_state.ss_data_ohlcv_display = df_to_filter[(df_to_filter.index.normalize() >= _start_dt_display_filter_pd.normalize()) & (df_to_filter.index.normalize() <= _end_dt_display_filter_pd.normalize())].copy()
            if st.session_state.ss_data_ohlcv_display.empty: st.warning(f"Nessun dato per display dopo filtraggio.")
            else: st.success(f"Dati per display pronti. Shape: {st.session_state.ss_data_ohlcv_display.shape}")
        elif st.session_state.ss_data_ohlcv_full is None: st.error(f"Fallimento caricamento storico completo.")
        
        if st.session_state.ss_data_ohlcv_full is not None and not st.session_state.ss_data_ohlcv_full.empty:
            progress_bar.progress(25, text="Calcolo feature...")
            st.session_state.ss_features_full = calculate_technical_features(st.session_state.ss_data_ohlcv_full)
            if st.session_state.ss_features_full.empty or len(st.session_state.ss_features_full) < 10:
                st.error("Fallimento calcolo feature o dati insuff.")
            else:
                st.success(f"Feature calcolate. Shape: {st.session_state.ss_features_full.shape}")
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
                            prediction_col_name_ml = f'prediction_{pred_horizon}d_pct_change'
                            st.session_state.ss_target_and_preds_full[prediction_col_name_ml] = predictions_series
                            df_ml_signals_full = generate_signals_from_ml_predictions(st.session_state.ss_target_and_preds_full, prediction_col_name_ml, CONFIG.get('signal_logic',{}).get('buy_threshold_change', 0.005), CONFIG.get('signal_logic',{}).get('sell_threshold_change', -0.005))
                            df_breakout_full = detect_breakout_signals(st.session_state.ss_features_full)
                            df_signals_combined_full = combine_signals(df_ml_signals_full, df_breakout_full)
                            df_signals_combined_full = apply_trading_spreads(df_signals_combined_full, current_asset_info_pipeline["type"], CONFIG.get('spreads',{}))
                            if not df_signals_combined_full.empty:
                                last_full_sig = df_signals_combined_full.iloc[-1]
                                asset_sym_curr_pipeline = current_asset_info_pipeline["symbol"]
                                if asset_sym_curr_pipeline in st.session_state.ss_asset_table_data:
                                    st.session_state.ss_asset_table_data[asset_sym_curr_pipeline]["ml_signal"] = last_full_sig.get('ml_signal', 'N/A')
                                    st.session_state.ss_asset_table_data[asset_sym_curr_pipeline]["breakout_signal"] = last_full_sig.get('breakout_signal', 'N/A')
                                    st.session_state.ss_asset_table_data[asset_sym_curr_pipeline]["last_price"] = f"{last_full_sig.get('Close', 0.0):.2f}" if 'Close' in last_full_sig else "N/A"
                            if st.session_state.ss_data_ohlcv_display is not None and not st.session_state.ss_data_ohlcv_display.empty:
                                common_idx_disp = st.session_state.ss_data_ohlcv_display.index.intersection(df_signals_combined_full.index)
                                if not common_idx_disp.empty:
                                    st.session_state.ss_final_signals_display = df_signals_combined_full.loc[common_idx_disp].copy()
                                    st.success(f"Segnali filtrati per display. Shape: {st.session_state.ss_final_signals_display.shape}")
                                    if not st.session_state.ss_final_signals_display.empty:
                                        last_disp_sig = st.session_state.ss_final_signals_display.iloc[-1]
                                        st.session_state.ss_last_signal_info_display = {"ticker": current_asset_info_pipeline["symbol"], "date": str(last_disp_sig.name), "ml_signal": last_disp_sig.get('ml_signal'), "breakout_signal": last_disp_sig.get('breakout_signal'), "close_price": f"{last_disp_sig.get('Close',0):.2f}"}
                                        if last_disp_sig.get('ml_signal') == 'BUY': play_buy_signal_sound(CONFIG.get('sound_utils',{}))
                                        elif last_disp_sig.get('ml_signal') == 'SELL': play_sell_signal_sound(CONFIG.get('sound_utils',{}))
        # --- FINE BLOCCO PIPELINE ---
        progress_bar.progress(100, text="Analisi completata!")
        time.sleep(0.5) 
        progress_bar.empty() 

    if st.session_state.get('ss_analysis_run_flag', False): 
        st.session_state.ss_analysis_run_flag = False
        logger.debug("Flag ss_analysis_run_flag resettato.")

# --- AREA PRINCIPALE PER VISUALIZZAZIONE RISULTATI ---
st.markdown("---")
asset_info_for_header = st.session_state.ss_current_asset_display_info
st.header(f"📊 Risultati per: {asset_info_for_header.get('name')} ({asset_info_for_header.get('symbol')}) - {asset_info_for_header.get('interval_label_short')}")

if st.session_state.ss_final_signals_display is not None and not st.session_state.ss_final_signals_display.empty:
    # ... (visualizzazione come prima) ...
    if st.session_state.ss_last_signal_info_display:
        st.subheader("📢 Ultimo Segnale (nell'intervallo visualizzato):")
        sig_info = st.session_state.ss_last_signal_info_display; ml_color = "green" if sig_info['ml_signal'] == "BUY" else "red" if sig_info['ml_signal'] == "SELL" else "gray"; breakout_color = "blue" if sig_info['breakout_signal'] == "BULLISH" else "orange" if sig_info['breakout_signal'] == "BEARISH" else "gray"
        st.markdown(f"*   **Data:** `{sig_info['date']}` *   **ML:** <span style='color:{ml_color};'>{sig_info['ml_signal']}</span> *   **Breakout:** <span style='color:{breakout_color};'>{sig_info['breakout_signal']}</span> *   **Prezzo:** `{sig_info['close_price']}`", unsafe_allow_html=True)
    st.markdown("---"); st.subheader("📈 Grafico Interattivo")
    df_features_for_chart = pd.DataFrame() 
    if st.session_state.ss_features_full is not None and not st.session_state.ss_features_full.empty and \
       st.session_state.ss_data_ohlcv_display is not None and not st.session_state.ss_data_ohlcv_display.empty:
        common_idx_chart = st.session_state.ss_data_ohlcv_display.index.intersection(st.session_state.ss_features_full.index)
        if not common_idx_chart.empty: df_features_for_chart = st.session_state.ss_features_full.loc[common_idx_chart].copy()
    if not df_features_for_chart.empty:
        chart_fig = create_main_stock_chart(df_features_for_chart, st.session_state.ss_final_signals_display, asset_info_for_header["symbol"], CONFIG.get('visualization',{}).get('show_ma_periods', [20, 50]))
        st.plotly_chart(chart_fig, use_container_width=True)
    else: 
        if st.session_state.get('ss_data_ohlcv_full') is not None: st.warning("Dati grafico insuff.")
    with st.expander("👁️ Dati Tabellari (ultimi 100 record dell'intervallo)"):
        if st.session_state.ss_data_ohlcv_display is not None and not st.session_state.ss_data_ohlcv_display.empty: st.dataframe(st.session_state.ss_data_ohlcv_display.tail(100))

elif 'ss_analysis_run_flag' in st.session_state and not st.session_state.ss_analysis_run_flag and asset_info_for_header.get('symbol'):
    st.info(f"Pronto per analizzare {asset_info_for_header['name']}. Clicca un intervallo nella tabella sopra.")
else: st.info("👋 Benvenuto! Seleziona un asset e un intervallo dalla tabella per iniziare.")

st.markdown("---")
st.caption(f"Dashboard v{APP_VERSION_FROM_CONFIG}.")
st.caption(f"Ora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}")
with st.expander("🔍 DEBUG: Stato Sessione Completo", expanded=False): 
    session_state_dict_for_json = {}
    for k, v_item in st.session_state.to_dict().items(): 
        if isinstance(v_item, pd.DataFrame): session_state_dict_for_json[k] = f"DataFrame shape {v_item.shape}" if v_item is not None else "None"
        elif isinstance(v_item, (datetime, date, pd.Timestamp, pd.Period)): session_state_dict_for_json[k] = str(v_item)
        else:
            try: json.dumps(v_item); session_state_dict_for_json[k] = v_item
            except (TypeError, OverflowError): session_state_dict_for_json[k] = str(v_item) 
    st.json(session_state_dict_for_json, expanded=False)
