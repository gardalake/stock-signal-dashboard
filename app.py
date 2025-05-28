# app.py - v1.6.9 (Daily ML Signal Logic, Display Granularity Variety)
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
    APP_VERSION_FROM_CONFIG = CONFIG.get('version', 'v1.6.9-daily-ml (config fallback)') 
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
# ... (Gestione altre chiavi API come prima)
LOADED_SECRETS = { "ALPHA_VANTAGE_API_KEY": ALPHA_VANTAGE_API_KEY, # ... e altre }

# --- DEFINIZIONI ASSET E INTERVALLI ---
TARGET_ASSETS_LIST = CONFIG.get('target_assets', [ 
    {"name": "Apple Inc.", "symbol": "AAPL", "type": "stock", "cg_id": None},
    {"name": "Microsoft Corp.", "symbol": "MSFT", "type": "stock", "cg_id": None},
    {"name": "Bitcoin", "symbol": "BTC", "type": "crypto", "cg_id": "bitcoin"},
    {"name": "Ethereum", "symbol": "ETH", "type": "crypto", "cg_id": "ethereum"},
]) # Lista ridotta per test più veloci, ripristina la tua completa
# Aggiungere asset di test se non presenti in config
if not any(a['symbol'] == "NONEXISTENT_STOCK" for a in TARGET_ASSETS_LIST): TARGET_ASSETS_LIST.append({"name": "Test Stock Inesistente", "symbol": "NONEXISTENT_STOCK", "type": "stock", "cg_id": None})
if not any(a.get('cg_id') == "nonexistent_crypto_id" for a in TARGET_ASSETS_LIST if a['type'] == 'crypto'): TARGET_ASSETS_LIST.append({"name": "Test Crypto Inesistente", "symbol": "NONEXISTENT_CRYPTO", "type": "crypto", "cg_id": "nonexistent_crypto_id"})

AV_DAILY_FUNC_NAME = CONFIG.get('alpha_vantage', {}).get('function', 'TIME_SERIES_DAILY_ADJUSTED')
# Etichetta bottone UI -> (codice_interno, av_interval_str, av_function_str, cg_days_for_granularity, is_intraday_flag, display_units_ago)
AVAILABLE_INTERVALS_ACTIONS = {
    "1H":  ("1H",   "60min", "TIME_SERIES_INTRADAY", 2,    True, 24), 
    "4H":  ("4H",   "60min", "TIME_SERIES_INTRADAY", 7,    True, 7*6), 
    "1G":  ("1D_D", "Daily", AV_DAILY_FUNC_NAME,     30,  False,30), 
    "1S":  ("1D_W", "Daily", AV_DAILY_FUNC_NAME,     7,   False,15), 
    "1M":  ("1D_M", "Daily", AV_DAILY_FUNC_NAME,     30,  False,12), 
}
DEFAULT_INTERVAL_BUTTON_LABEL = "1G" 

# --- STATO DELLA SESSIONE ---
# ... (Inizializzazione stato sessione come prima, assicurandosi che i DataFrame siano None)
# Nuova variabile per i dati giornalieri per ML
if 'ss_data_ohlcv_daily_for_ml' not in st.session_state:
    st.session_state.ss_data_ohlcv_daily_for_ml = None
# ... (altre inizializzazioni come prima)
if 'ss_current_asset_display_info' not in st.session_state: 
    first_asset = TARGET_ASSETS_LIST[0] if TARGET_ASSETS_LIST else {}
    default_symbol = first_asset.get("symbol","N/A")
    default_interval_label_for_state = DEFAULT_INTERVAL_BUTTON_LABEL if DEFAULT_INTERVAL_BUTTON_LABEL in AVAILABLE_INTERVALS_ACTIONS else list(AVAILABLE_INTERVALS_ACTIONS.keys())[0]
    st.session_state.ss_current_asset_display_info = {
        "name": first_asset.get("name","N/A"), "symbol": default_symbol, "type": first_asset.get("type","N/A"), 
        "cg_id": first_asset.get("cg_id"), "interval_code": AVAILABLE_INTERVALS_ACTIONS[default_interval_label_for_state][0], 
        "interval_label_short": default_interval_label_for_state 
    }
if 'ss_asset_table_data' not in st.session_state:
    st.session_state.ss_asset_table_data = { asset["symbol"]: {**asset, "last_price": "N/A", "ml_signal": "N/A", "breakout_signal": "N/A"} for asset in TARGET_ASSETS_LIST }
for key in ['ss_data_ohlcv_full', 'ss_data_ohlcv_display', 'ss_features_full', 'ss_target_and_preds_full', 'ss_final_signals_display', 'ss_trained_ml_model', 'ss_last_signal_info_display', 'ss_analysis_run_flag']:
    if key not in st.session_state: st.session_state[key] = None if "df" in key or "model" in key else False


# --- UI PRINCIPALE ---
st.title(f"📊 Asset Signal Dashboard"); st.caption(f"Versione: {APP_VERSION_FROM_CONFIG}"); st.markdown("---")
api_warning_placeholder_main = st.empty()
# ... (Tabella Asset come prima, con bottoni) ...
st.subheader("📈 Asset Overview & Analysis Triggers")
col_proportions = [0.22, 0.08, 0.12, 0.12, 0.12, 0.07, 0.07, 0.07, 0.07, 0.07] 
header_cols = st.columns(col_proportions)
headers = ["Nome", "Simbolo", "Prezzo", "Segnale ML", "Breakout", "1H", "4H", "1G", "1S", "1M"]
for col, header_text in zip(header_cols, headers): col.markdown(f"**{header_text}**")
st.markdown("<hr style='margin-top:0.5rem; margin-bottom:0.5rem;'>", unsafe_allow_html=True)

for asset_symbol_key in [a["symbol"] for a in TARGET_ASSETS_LIST]:
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
        if 5 + i < len(row_cols):
            interval_details_tuple_btn = AVAILABLE_INTERVALS_ACTIONS[short_label]
            interval_code_for_button = interval_details_tuple_btn[0]
            button_key = f"btn_{asset_symbol_key}_{short_label}"
            is_selected_button = (st.session_state.ss_current_asset_display_info["symbol"] == asset_symbol_key and st.session_state.ss_current_asset_display_info["interval_label_short"] == short_label)
            button_type = "primary" if is_selected_button else "secondary"
            if row_cols[5+i].button(short_label, key=button_key, use_container_width=True, type=button_type):
                st.session_state.ss_current_asset_display_info = {"name": asset_static_info["name"], "symbol": asset_symbol_key, "type": asset_static_info["type"], "cg_id": asset_static_info.get("cg_id"), "interval_code": interval_code_for_button, "interval_label_short": short_label }
                st.session_state.ss_analysis_run_flag = True 
                st.session_state.ss_data_ohlcv_full = None; st.session_state.ss_data_ohlcv_display = None; st.session_state.ss_features_full = None; st.session_state.ss_target_and_preds_full = None; st.session_state.ss_final_signals_display = None; st.session_state.ss_trained_ml_model = None; st.session_state.ss_data_ohlcv_daily_for_ml = None # Resetta anche questo
                logger.info(f"Analisi richiesta per {asset_symbol_key}, intervallo {short_label} ({interval_code_for_button})")
                st.rerun() 
    st.markdown("<hr style='margin-top:0.2rem; margin-bottom:0.2rem;'>", unsafe_allow_html=True)
st.markdown("---") 

# --- PIPELINE DI ELABORAZIONE ---
if st.session_state.get('ss_analysis_run_flag', False):
    # Definizioni parametri API basate sull'intervallo SELEZIONATO PER IL DISPLAY
    current_asset_info_pipeline = st.session_state.ss_current_asset_display_info
    current_interval_label_short_pipeline = current_asset_info_pipeline["interval_label_short"] 
    interval_details_pipeline = AVAILABLE_INTERVALS_ACTIONS.get(current_interval_label_short_pipeline)
    if not interval_details_pipeline:
        interval_details_pipeline = AVAILABLE_INTERVALS_ACTIONS[DEFAULT_INTERVAL_BUTTON_LABEL]
    
    interval_code_display, av_api_interval_display, av_api_function_display, cg_api_days_granularity_display, interval_is_intraday_display, display_units_ago_display = interval_details_pipeline

    # Calcola date per VISUALIZZAZIONE
    _display_end_date_dt_p = datetime.now() 
    # ... (logica _display_start_date_dt_p come prima, usando _display vars)
    if interval_code_display == "1H": _display_start_date_dt_p = _display_end_date_dt_p - timedelta(hours=display_units_ago_display)
    elif interval_code_display == "4H": _display_start_date_dt_p = _display_end_date_dt_p - timedelta(hours=display_units_ago_display * 4) 
    elif interval_code_display.startswith("1D"): 
        time_unit_char = interval_code_display.split('_')[-1][-1]; num_part_str = interval_code_display.split('_')[-1][:-1] if len(interval_code_display.split('_')[-1]) > 1 else str(display_units_ago_display); num_val = int(num_part_str) if num_part_str.isdigit() else display_units_ago_display
        if time_unit_char == "W": _display_start_date_dt_p = _display_end_date_dt_p - timedelta(weeks=num_val)
        elif time_unit_char == "M": _display_start_date_dt_p = _display_end_date_dt_p - timedelta(days=num_val * 30) 
        elif time_unit_char == "Y": _display_start_date_dt_p = _display_end_date_dt_p - timedelta(days=num_val * 365)
        else: _display_start_date_dt_p = _display_end_date_dt_p - timedelta(days=num_val) 
    else: _display_start_date_dt_p = _display_end_date_dt_p - timedelta(days=30) 

    # Parametri per caricare DATI GIORNALIERI PER ML
    MIN_DAYS_FOR_ML_p = CONFIG.get('ml_model', {}).get('min_days_for_indicators_and_training', 200)
    _ml_data_load_end_date_dt_p = date.today() # Carica dati giornalieri fino a ieri/oggi
    _ml_data_load_start_date_dt_p = _ml_data_load_end_date_dt_p - timedelta(days=MIN_DAYS_FOR_ML_p -1)
    _av_outputsize_ml_p = "full" # Sempre full per dati giornalieri ML
    _cg_days_to_fetch_ml_p = (date.today() - _ml_data_load_start_date_dt_p).days + 1
    if _cg_days_to_fetch_ml_p <= 0: _cg_days_to_fetch_ml_p = MIN_DAYS_FOR_ML_p

    log_container = st.container()
    with log_container:
        st.markdown(f"### ⚙️ Analisi per: {current_asset_info_pipeline['name']} ({current_asset_info_pipeline['symbol']}) - Visualizzazione: {current_asset_info_pipeline['interval_label_short']}")
        progress_bar = st.progress(0, text="Inizio...")
        
        # 1A. CARICAMENTO DATI PER DISPLAY (granularità variabile)
        progress_bar.progress(5, text=f"Caricamento dati display ({current_asset_info_pipeline['interval_label_short']})...")
        asset_to_fetch = current_asset_info_pipeline
        if asset_to_fetch["type"] == "stock":
            if not ALPHA_VANTAGE_API_KEY: st.error("Chiave API AV mancante."); st.session_state.ss_data_ohlcv_display = None
            else:
                av_call_params_display = {}
                if av_api_function_display == "TIME_SERIES_INTRADAY": av_call_params_display['av_interval'] = av_api_interval_display
                # Usa outputsize specifico per intraday display (es. compact) se necessario
                outputsize_display = "compact" if interval_is_intraday_display else "full"
                st.session_state.ss_data_ohlcv_display = get_stock_data(ALPHA_VANTAGE_API_KEY, asset_to_fetch["symbol"], av_api_function_display, outputsize_display, **av_call_params_display)
        elif asset_to_fetch["type"] == "crypto":
            days_for_cg_display = (date.today() - _display_start_date_dt_p.date()).days + 1 if not interval_is_intraday_display else cg_api_days_granularity_display
            if days_for_cg_display <=0 : days_for_cg_display = cg_api_days_granularity_display if interval_is_intraday_display else 30
            logger.debug(f"Caricamento crypto display - Giorni: {days_for_cg_display}, intervallo: {interval_code_display}")
            st.session_state.ss_data_ohlcv_display = get_crypto_data(asset_to_fetch.get("cg_id"), CONFIG.get('coingecko',{}).get('vs_currency', 'usd'), days_for_cg_display, interval_code_display)

        if st.session_state.ss_data_ohlcv_display is not None and not st.session_state.ss_data_ohlcv_display.empty:
            # Filtra ulteriormente i dati display se caricato uno storico più ampio del necessario per la granularità
            _start_dt_disp_filt_pd = pd.to_datetime(_display_start_date_dt_p); _end_dt_disp_filt_pd = pd.to_datetime(_display_end_date_dt_p)
            if not isinstance(st.session_state.ss_data_ohlcv_display.index, pd.DatetimeIndex): st.session_state.ss_data_ohlcv_display.index = pd.to_datetime(st.session_state.ss_data_ohlcv_display.index)
            df_to_filt_disp = st.session_state.ss_data_ohlcv_display
            if interval_is_intraday_p: st.session_state.ss_data_ohlcv_display = df_to_filt_disp[(df_to_filt_disp.index >= _start_dt_disp_filt_pd) & (df_to_filt_disp.index < _end_dt_disp_filt_pd + pd.Timedelta(days=1))].copy()
            else: st.session_state.ss_data_ohlcv_display = df_to_filt_disp[(df_to_filt_disp.index.normalize() >= _start_dt_disp_filt_pd.normalize()) & (df_to_filt_disp.index.normalize() <= _end_dt_disp_filt_pd.normalize())].copy()
            if st.session_state.ss_data_ohlcv_display.empty: st.warning(f"Nessun dato per display dopo filtraggio.")
            else: st.success(f"Dati per display pronti. Shape: {st.session_state.ss_data_ohlcv_display.shape}")
        else: st.error(f"Fallimento caricamento dati display per {asset_to_fetch['symbol']}.")


        # 1B. CARICAMENTO DATI GIORNALIERI PER ML
        progress_bar.progress(15, text=f"Caricamento dati giornalieri per ML ({asset_to_fetch['symbol']})...")
        if asset_to_fetch["type"] == "stock":
            if not ALPHA_VANTAGE_API_KEY: st.error("Chiave API AV mancante per dati ML."); st.session_state.ss_data_ohlcv_daily_for_ml = None
            else: st.session_state.ss_data_ohlcv_daily_for_ml = get_stock_data(ALPHA_VANTAGE_API_KEY, asset_to_fetch["symbol"], AV_DAILY_FUNC_NAME, "full")
        elif asset_to_fetch["type"] == "crypto":
            st.session_state.ss_data_ohlcv_daily_for_ml = get_crypto_data(asset_to_fetch.get("cg_id"), CONFIG.get('coingecko',{}).get('vs_currency', 'usd'), _cg_days_to_fetch_ml_p, "1D_D") # Forza richiesta giornaliera

        if st.session_state.ss_data_ohlcv_daily_for_ml is not None and not st.session_state.ss_data_ohlcv_daily_for_ml.empty:
            logger.info(f"Dati giornalieri per ML caricati. Shape: {st.session_state.ss_data_ohlcv_daily_for_ml.shape}")
            # Filtra per assicurarsi che non ci siano dati troppo futuri se _ml_data_load_end_date_dt_p è oggi
            st.session_state.ss_data_ohlcv_daily_for_ml = st.session_state.ss_data_ohlcv_daily_for_ml[st.session_state.ss_data_ohlcv_daily_for_ml.index.normalize() <= pd.to_datetime(_ml_data_load_end_date_dt_p).normalize()].copy()
        else:
            st.error(f"Fallimento caricamento dati giornalieri per ML ({asset_to_fetch['symbol']}). Pipeline ML non può procedere.")
            st.session_state.ss_analysis_run_flag = False; progress_bar.empty(); st.stop()

        # --- INIZIO ELABORAZIONE ML (usa ss_data_ohlcv_daily_for_ml) ---
        if st.session_state.ss_data_ohlcv_daily_for_ml is not None and not st.session_state.ss_data_ohlcv_daily_for_ml.empty:
            # ... (Pipeline ML come prima, ma ora usa ss_data_ohlcv_daily_for_ml come input)
            # Rinominare le variabili interne alla pipeline per chiarezza, es. df_ml_input = st.session_state.ss_data_ohlcv_daily_for_ml
            df_ml_input = st.session_state.ss_data_ohlcv_daily_for_ml
            progress_bar.progress(25, text="Calcolo feature ML (su dati giornalieri)...")
            features_for_ml = calculate_technical_features(df_ml_input) # Questo ora è ss_features_full
            st.session_state.ss_features_full = features_for_ml # Salva le feature giornaliere

            if features_for_ml.empty or len(features_for_ml) < 10: st.error("Fallimento calcolo feature ML o dati insuff.")
            else: # Prosegui con ML
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
                            st.session_state.ss_target_and_preds_full = df_target_ml.copy() # Questo è giornaliero
                            pred_col_name_ml = f'prediction_{pred_horizon_days_ml}d_pct_change'
                            st.session_state.ss_target_and_preds_full[pred_col_name_ml] = preds_series_ml
                            
                            # Genera segnali ML (giornalieri)
                            ml_signals_df_full = generate_signals_from_ml_predictions(st.session_state.ss_target_and_preds_full, pred_col_name_ml, CONFIG.get('signal_logic',{}).get('buy_threshold_change', 0.005), CONFIG.get('signal_logic',{}).get('sell_threshold_change', -0.005))
                            # Breakout (calcolati anche su dati giornalieri, da ss_features_full)
                            breakout_df_full = detect_breakout_signals(features_for_ml) # features_for_ml è giornaliero
                            
                            signals_combined_daily = combine_signals(ml_signals_df_full, breakout_df_full)
                            signals_combined_daily = apply_trading_spreads(signals_combined_daily, asset_to_fetch["type"], CONFIG.get('spreads',{}))

                            # Aggiorna la tabella UI con l'ULTIMO segnale GIORNALIERO
                            if not signals_combined_daily.empty:
                                last_daily_signal_row = signals_combined_daily.iloc[-1]
                                asset_sym_update = asset_to_fetch["symbol"]
                                if asset_sym_update in st.session_state.ss_asset_table_data:
                                    st.session_state.ss_asset_table_data[asset_sym_update]["ml_signal"] = last_daily_signal_row.get('ml_signal', 'N/A')
                                    st.session_state.ss_asset_table_data[asset_sym_update]["breakout_signal"] = last_daily_signal_row.get('breakout_signal', 'N/A')
                                    st.session_state.ss_asset_table_data[asset_sym_update]["last_price"] = f"{last_daily_signal_row.get('Close', 0.0):.2f}" if 'Close' in last_daily_signal_row else "N/A"
                            
                            # Per la visualizzazione, dobbiamo allineare i segnali giornalieri con l'indice dei dati di display (che potrebbe essere intraday)
                            # Questo significa che un segnale giornaliero si applica a tutte le candele di quel giorno.
                            if st.session_state.ss_data_ohlcv_display is not None and not st.session_state.ss_data_ohlcv_display.empty:
                                # Prendi solo le colonne di segnale da signals_combined_daily
                                signals_to_merge = signals_combined_daily[['ml_signal', 'breakout_signal']].copy()
                                # Assicura che l'indice di signals_to_merge sia solo date (per il merge con intraday)
                                signals_to_merge.index = signals_to_merge.index.normalize()
                                
                                # Prepara il df di display con un indice normalizzato per il merge
                                df_display_with_date_idx = st.session_state.ss_data_ohlcv_display.copy()
                                df_display_with_date_idx['temp_date_idx'] = df_display_with_date_idx.index.normalize()
                                
                                # Merge basato sulla data normalizzata
                                merged_for_display = pd.merge(
                                    df_display_with_date_idx, 
                                    signals_to_merge, 
                                    left_on='temp_date_idx', 
                                    right_index=True, 
                                    how='left'
                                )
                                merged_for_display.drop(columns=['temp_date_idx'], inplace=True)
                                # Propaga i segnali giornalieri alle candele intraday (ffill)
                                merged_for_display[['ml_signal', 'breakout_signal']] = merged_for_display[['ml_signal', 'breakout_signal']].fillna(method='ffill')
                                st.session_state.ss_final_signals_display = merged_for_display.copy()
                                
                                st.success(f"Segnali (giornalieri) allineati ai dati display. Shape: {st.session_state.ss_final_signals_display.shape}")

                                if not st.session_state.ss_final_signals_display.empty:
                                    last_disp_sig = st.session_state.ss_final_signals_display.iloc[-1] # Ultimo segnale VISUALIZZATO
                                    st.session_state.ss_last_signal_info_display = { # Info per il riquadro "Ultimo Segnale"
                                        "ticker": asset_to_fetch["symbol"], 
                                        "date": last_disp_sig.name.strftime('%Y-%m-%d %H:%M:%S') if isinstance(last_disp_sig.name, pd.Timestamp) else str(last_disp_sig.name), 
                                        "ml_signal": last_daily_signal_row.get('ml_signal'), # Mostra l'ULTIMO segnale GIORNALIERO
                                        "breakout_signal": last_daily_signal_row.get('breakout_signal'), # idem
                                        "close_price": f"{last_disp_sig.get('Close',0):.2f}" # Prezzo della candela display
                                    }
                                    # Suoni/Email basati sull'ULTIMO segnale GIORNALIERO
                                    if last_daily_signal_row.get('ml_signal') == 'BUY': play_buy_signal_sound(CONFIG.get('sound_utils',{}))
                                    elif last_daily_signal_row.get('ml_signal') == 'SELL': play_sell_signal_sound(CONFIG.get('sound_utils',{}))
        # --- FINE BLOCCO PIPELINE ---
        progress_bar.progress(100, text="Analisi completata!")
        time.sleep(0.5); progress_bar.empty() 
    if st.session_state.get('ss_analysis_run_flag', False): st.session_state.ss_analysis_run_flag = False

# --- AREA PRINCIPALE PER VISUALIZZAZIONE RISULTATI ---
st.markdown("---")
asset_info_for_header = st.session_state.ss_current_asset_display_info # Questo è sempre definito
st.header(f"📊 Risultati per: {asset_info_for_header.get('name')} ({asset_info_for_header.get('symbol')}) - Intervallo Display: {asset_info_for_header.get('interval_label_short')}")

# Mostra warning AV se necessario
if asset_info_for_header.get("type") == "stock" and not ALPHA_VANTAGE_API_KEY and CONFIG.get('alpha_vantage'): 
    st.warning("Chiave API Alpha Vantage non configurata. I dati per le azioni potrebbero non essere disponibili o limitati.")


if st.session_state.get('ss_final_signals_display') is not None and isinstance(st.session_state.ss_final_signals_display, pd.DataFrame) and not st.session_state.ss_final_signals_display.empty:
    if st.session_state.ss_last_signal_info_display:
        st.subheader("📢 Ultimo Segnale Calcolato (Giornaliero):") # Modificato per chiarezza
        sig_info = st.session_state.ss_last_signal_info_display
        ml_color = "green" if sig_info['ml_signal'] == "BUY" else "red" if sig_info['ml_signal'] == "SELL" else "gray"; breakout_color = "blue" if sig_info['breakout_signal'] == "BULLISH" else "orange" if sig_info['breakout_signal'] == "BEARISH" else "gray"
        st.markdown(f"*   **Data Segnale (Giornaliero):** `{sig_info['date'].split(' ')[0]}` *   **Segnale ML:** <span style='color:{ml_color};'>{sig_info['ml_signal']}</span> *   **Breakout:** <span style='color:{breakout_color};'>{sig_info['breakout_signal']}</span> *   **Prezzo Chiusura (Candela Display):** `{sig_info['close_price']}`", unsafe_allow_html=True)
    st.markdown("---"); st.subheader("📈 Grafico Interattivo")
    
    # Per il grafico, usiamo ss_data_ohlcv_display (che ha la granularità scelta)
    # e ss_final_signals_display (che ha i segnali giornalieri propagati/allineati)
    df_for_chart_ohlcv = st.session_state.ss_data_ohlcv_display
    # Le feature (MA, Bollinger) dovrebbero essere calcolate sulla granularità del display per essere significative sul grafico
    # Questo è un punto da migliorare: ora ss_features_full è sempre giornaliero.
    # Per ora, passiamo ss_data_ohlcv_display e la funzione grafico aggiungerà MA se le trova.
    # Idealmente, dovremmo ricalcolare le MA per il display.
    # Per semplicità immediata, il grafico userà le MA da ss_features_full se gli indici combaciano, altrimenti non le mostrerà.
    
    # Soluzione temporanea: se ss_features_full esiste e copre l'intervallo di display, usalo per le MA
    df_ma_for_chart = pd.DataFrame()
    if st.session_state.ss_features_full is not None and not st.session_state.ss_features_full.empty and \
       df_for_chart_ohlcv is not None and not df_for_chart_ohlcv.empty:
        common_idx_ma_chart = df_for_chart_ohlcv.index.intersection(st.session_state.ss_features_full.index)
        if not common_idx_ma_chart.empty:
            df_ma_for_chart = st.session_state.ss_features_full.loc[common_idx_ma_chart].copy()

    # Unisci df_for_chart_ohlcv (che ha OHLCV per display) con df_ma_for_chart (che ha MA per display)
    df_chart_input = df_for_chart_ohlcv.copy()
    if not df_ma_for_chart.empty:
        ma_cols_to_add_to_chart = [col for col in CONFIG.get('visualization',{}).get('show_ma_periods', ['MA20', 'MA50']) if col in df_ma_for_chart.columns]
        if ma_cols_to_add_to_chart: # Aggiungi solo se le colonne MA esistono
             df_chart_input = pd.merge(df_chart_input, df_ma_for_chart[ma_cols_to_add_to_chart], left_index=True, right_index=True, how='left')


    if not df_chart_input.empty:
        chart_fig = create_main_stock_chart(
            df_ohlcv_ma=df_chart_input, # Contiene OHLCV display + MA (se allineate)
            df_signals=st.session_state.ss_final_signals_display, # Contiene segnali giornalieri allineati a display
            ticker=asset_info_for_header["symbol"],
            ma_periods_to_show=CONFIG.get('visualization',{}).get('show_ma_periods', [20, 50])
        )
        st.plotly_chart(chart_fig, use_container_width=True)
    else: 
        if st.session_state.get('ss_data_ohlcv_full') is not None: st.warning("Dati grafico insuff.")

    with st.expander("👁️ Dati Tabellari (ultimi 100 record dell'intervallo visualizzato)"):
        if st.session_state.ss_data_ohlcv_display is not None and not st.session_state.ss_data_ohlcv_display.empty: 
            st.markdown("#### Dati Grezzi (per Display)")
            st.dataframe(st.session_state.ss_data_ohlcv_display.tail(100))
        # Mostra ss_features_full (giornaliero) e ss_target_and_preds_full (giornaliero) senza filtraggio per ora,
        # o potremmo filtrarli per le date corrispondenti al display se ha senso.
        if st.session_state.ss_features_full is not None and not st.session_state.ss_features_full.empty:
            st.markdown("#### Feature Tecniche (Base Giornaliera per ML)")
            st.dataframe(st.session_state.ss_features_full.tail(100))
        if st.session_state.ss_target_and_preds_full is not None and not st.session_state.ss_target_and_preds_full.empty:
            st.markdown("#### Target & Predizioni ML (Base Giornaliera)")
            st.dataframe(st.session_state.ss_target_and_preds_full.tail(100))
        if st.session_state.ss_final_signals_display is not None and not st.session_state.ss_final_signals_display.empty: 
            st.markdown("#### Segnali Finali (Allineati al Display)")
            st.dataframe(st.session_state.ss_final_signals_display.tail(100))

elif 'ss_analysis_run_flag' in st.session_state and not st.session_state.ss_analysis_run_flag and asset_info_for_header.get('symbol'):
    st.info(f"Pronto per analizzare {asset_info_for_header['name']}. Clicca un intervallo nella tabella sopra.")
else: st.info("👋 Benvenuto! Seleziona un asset e un intervallo dalla tabella per iniziare.")

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
