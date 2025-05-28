# app.py - v1.6.7 (UI Refactor - Asset Table Dashboard)
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
except FileNotFoundError: #etc.
    # ... (gestione errori config come prima) ...
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

# Gestione errori config (dopo set_page_config)
# ... (come prima) ...
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
# ... (resto gestione chiavi API come prima, ma i warning andranno nel corpo principale se necessario)
LOADED_SECRETS = { "ALPHA_VANTAGE_API_KEY": ALPHA_VANTAGE_API_KEY, # ... e altre }

# --- DEFINIZIONI ASSET E INTERVALLI ---
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

# Intervalli: Etichetta per UI -> (codice_interno, av_interval_str, av_function_str, cg_days_for_granularity, is_intraday_flag, display_units_ago)
# display_units_ago: quante unità di questo intervallo mostrare nel grafico (es. 30 ore, 30 settimane)
# cg_days_for_granularity è usato per ottenere la granularità desiderata, non per l'estensione del grafico.
AVAILABLE_INTERVALS_ACTIONS = {
    "1H":  ("1H",   "60min", "TIME_SERIES_INTRADAY", 1,   True, 30), # Mostra ultime 30 ore
    "4H":  ("4H",   "60min", "TIME_SERIES_INTRADAY", 7,   True, 30), # Mostra ultime 30 "4-ore" (aggregare da 60min)
    "1G":  ("1D",   "Daily", CONFIG.get('alpha_vantage', {}).get('function', 'TIME_SERIES_DAILY_ADJUSTED'), 30,  False,30), # Mostra ultimi 30 giorni
    "1S":  ("1W",   "Daily", CONFIG.get('alpha_vantage', {}).get('function', 'TIME_SERIES_DAILY_ADJUSTED'), 30*7, False,30), # Mostra ultime 30 settimane
    "1M":  ("1M",   "Daily", CONFIG.get('alpha_vantage', {}).get('function', 'TIME_SERIES_DAILY_ADJUSTED'), 30*12,False,30), # Mostra ultimi 30 mesi
}

# --- STATO DELLA SESSIONE ---
if 'ss_current_asset_display_info' not in st.session_state: # Info sull'asset attualmente nel grafico
    st.session_state.ss_current_asset_display_info = {
        "name": TARGET_ASSETS_LIST[0]["name"],
        "symbol": TARGET_ASSETS_LIST[0]["symbol"],
        "type": TARGET_ASSETS_LIST[0]["type"],
        "cg_id": TARGET_ASSETS_LIST[0]["cg_id"],
        "interval_code": AVAILABLE_INTERVALS_ACTIONS["1G"][0], # Default a giornaliero
        "interval_label_short": "1G" 
    }
# Dati per la tabella (memorizza l'ultimo stato noto per ogni asset)
if 'ss_asset_table_data' not in st.session_state:
    st.session_state.ss_asset_table_data = {
        asset["symbol"]: asset.copy() for asset in TARGET_ASSETS_LIST
    }
# Altri stati necessari per la pipeline di analisi (simili a prima)
for key in ['ss_data_ohlcv_full', 'ss_data_ohlcv_display', 'ss_features_full', 
            'ss_target_and_preds_full', 'ss_final_signals_display', 
            'ss_trained_ml_model', 'ss_analysis_run_flag']:
    if key not in st.session_state:
        st.session_state[key] = None if "df" in key or "model" in key else False


# --- TITOLO E HEADER ---
st.title(f"📊 Asset Signal Dashboard")
st.caption(f"Versione: {APP_VERSION_FROM_CONFIG}")
st.markdown("---")

# Placeholder per warning API Key
api_warning_placeholder_main = st.empty()
if not ALPHA_VANTAGE_API_KEY and CONFIG.get('alpha_vantage') and st.session_state.ss_current_asset_display_info["type"] == "stock":
    api_warning_placeholder_main.warning("Chiave API Alpha Vantage non configurata. I dati per le azioni non saranno disponibili.")
else:
    api_warning_placeholder_main.empty()


# --- TABELLA ASSET E CONTROLLI INTERVALLO ---
st.subheader("📈 Asset Overview & Analysis Trigger")

# Header Tabella
cols_header = st.columns([0.2, 0.1, 0.15, 0.15, 0.15, 0.07, 0.07, 0.07, 0.07, 0.07])
headers = ["Nome", "Simbolo", "Prezzo", "Segnale ML", "Breakout", "1H", "4H", "1G", "1S", "1M"]
for col, header_text in zip(cols_header, headers):
    col.markdown(f"**{header_text}**")

# Righe Tabella Dinamiche
for asset_symbol_key, asset_data_in_state in st.session_state.ss_asset_table_data.items():
    # Trova l'asset originale dalla lista per i dettagli statici (nome, tipo, cg_id)
    asset_static_info = next((a for a in TARGET_ASSETS_LIST if a["symbol"] == asset_symbol_key), None)
    if not asset_static_info: continue

    cols_row = st.columns([0.2, 0.1, 0.15, 0.15, 0.15, 0.07, 0.07, 0.07, 0.07, 0.07])
    cols_row[0].markdown(asset_static_info["name"])
    cols_row[1].markdown(f"`{asset_static_info['symbol']}`")
    cols_row[2].markdown(asset_data_in_state.get("last_price", "N/A")) # Prende da session_state
    
    ml_signal = asset_data_in_state.get("ml_signal", "N/A")
    ml_color = "green" if ml_signal == "BUY" else "red" if ml_signal == "SELL" else "gray"
    cols_row[3].markdown(f"<span style='color:{ml_color};'>{ml_signal}</span>", unsafe_allow_html=True)

    breakout_signal = asset_data_in_state.get("breakout_signal", "N/A")
    breakout_color = "blue" if breakout_signal == "BULLISH" else "orange" if breakout_signal == "BEARISH" else "gray"
    cols_row[4].markdown(f"<span style='color:{breakout_color};'>{breakout_signal}</span>", unsafe_allow_html=True)

    # Bottoni Intervallo
    interval_buttons_cols = [cols_row[5], cols_row[6], cols_row[7], cols_row[8], cols_row[9]]
    for i, (interval_label_short, (interval_code, _, _, _, _, _)) in enumerate(AVAILABLE_INTERVALS_ACTIONS.items()):
        button_key = f"btn_{asset_static_info['symbol']}_{interval_label_short}"
        if interval_buttons_cols[i].button(interval_label_short, key=button_key, use_container_width=True):
            st.session_state.ss_current_asset_display_info = { # Aggiorna l'asset e l'intervallo da visualizzare
                "name": asset_static_info["name"],
                "symbol": asset_static_info["symbol"],
                "type": asset_static_info["type"],
                "cg_id": asset_static_info["cg_id"],
                "interval_code": interval_code, # Codice interno es. "1H", "1D"
                "interval_label_short": interval_label_short # Etichetta breve es "1H"
            }
            st.session_state.ss_analysis_run_flag = True # Attiva l'analisi
            # Resetta i dati precedenti per la nuova analisi
            st.session_state.ss_data_ohlcv_full = None
            st.session_state.ss_data_ohlcv_display = None
            # ... (reset altri stati relativi ai dati e al modello)
            logger.info(f"Analisi richiesta per {asset_static_info['symbol']} con intervallo {interval_label_short} ({interval_code})")
            st.experimental_rerun() # Forzare il re-run per avviare subito la pipeline con il nuovo stato

st.markdown("---")

# --- LOGICA DI CALCOLO START/END DATE PER API E DISPLAY ---
# Basata su ss_current_asset_display_info (l'asset e intervallo attualmente selezionato per il grafico)
current_asset_info = st.session_state.ss_current_asset_display_info
current_interval_code = current_asset_info["interval_code"]
interval_details_tuple = next((v for k,v in AVAILABLE_INTERVALS_ACTIONS.items() if v[0] == current_interval_code), AVAILABLE_INTERVALS_ACTIONS["1G"]) # Fallback a 1G

_, av_api_interval, av_api_function, cg_api_days_granularity, interval_is_intraday, display_units_ago = interval_details_tuple

# Calcola date per la VISUALIZZAZIONE del grafico
_display_end_date_dt = datetime.now() # Fine visualizzazione è "ora" (per intraday) o fine giornata per daily
_display_start_date_dt = None

if current_interval_code == "1H": _display_start_date_dt = _display_end_date_dt - timedelta(hours=display_units_ago)
elif current_interval_code == "4H": _display_start_date_dt = _display_end_date_dt - timedelta(hours=display_units_ago * 4)
elif current_interval_code == "1G": _display_start_date_dt = _display_end_date_dt - timedelta(days=display_units_ago)
elif current_interval_code == "1S": _display_start_date_dt = _display_end_date_dt - timedelta(weeks=display_units_ago)
elif current_interval_code == "1M": _display_start_date_dt = _display_end_date_dt - timedelta(days=display_units_ago * 30) # Approssimazione
else: _display_start_date_dt = _display_end_date_dt - timedelta(days=30) # Fallback

# Calcola date per il CARICAMENTO DATI (storico più lungo per ML/TA)
MIN_PERIODS_FOR_ML_TA = CONFIG.get('ml_model', {}).get('min_days_for_indicators_and_training', 200) # Questo è in "periodi" della granularità principale (giornaliera)
# Se stiamo caricando dati intraday, MIN_PERIODS_FOR_ML_TA deve essere convertito in un numero di giorni appropriato.
# Esempio: se min è 200 giorni, e stiamo caricando dati orari, potremmo voler caricare solo (es.) gli ultimi 30 giorni di dati orari.
# Questa logica va raffinata. Per ora, per intraday, carichiamo un numero fisso di giorni per la granularità.
_api_data_load_start_date_dt = _display_start_date_dt - timedelta(days=MIN_PERIODS_FOR_ML_TA if not interval_is_intraday else 30) # Carica 30gg di storico per intraday, 200gg per daily
_api_data_load_end_date_dt = _display_end_date_dt # Carica fino ad oggi

_av_outputsize_param = "compact" if interval_is_intraday and current_asset_info["type"] == "stock" else "full"

_cg_days_to_fetch_param = (date.today() - _api_data_load_start_date_dt.date()).days + 1
if _cg_days_to_fetch_param <= 0: _cg_days_to_fetch_param = MIN_PERIODS_FOR_ML_TA if not interval_is_intraday else 30
if interval_is_intraday and current_asset_info["type"] == "crypto":
    _cg_days_to_fetch_param = cg_api_days_granularity # Usa i giorni specifici per ottenere la granularità da CG

# --- PIPELINE DI ELABORAZIONE DATI E SEGNALI ---
if st.session_state.get('ss_analysis_run_flag', False): # Usa .get per sicurezza
    log_container = st.container()
    with log_container:
        st.markdown(f"### ⚙️ Analisi per: {current_asset_info['name']} ({current_asset_info['symbol']}) - Intervallo: {current_asset_info['interval_label_short']}")
        progress_bar = st.progress(0, text="Inizio analisi...")
        
        logger.info(f"Inizio caricamento. Display: {_display_start_date_dt.strftime('%Y-%m-%d %H:%M')} a {_display_end_date_dt.strftime('%Y-%m-%d %H:%M')}. API Load Start: {_api_data_load_start_date_dt.strftime('%Y-%m-%d')}")
        progress_bar.progress(10, text=f"Caricamento storico per {current_asset_info['symbol']}...")
        
        # 1. CARICAMENTO DATI
        if current_asset_info["type"] == "stock":
            # ... (logica get_stock_data come prima, usando av_api_function, _av_outputsize_param, e av_api_interval)
            av_call_params = {}
            if av_api_function == "TIME_SERIES_INTRADAY": av_call_params['av_interval'] = av_api_interval
            st.session_state.ss_data_ohlcv_full = get_stock_data(ALPHA_VANTAGE_API_KEY, current_asset_info["symbol"], av_api_function, _av_outputsize_param, **av_call_params)
        elif current_asset_info["type"] == "crypto":
            # ... (logica get_crypto_data come prima, usando _cg_days_to_fetch_param)
            st.session_state.ss_data_ohlcv_full = get_crypto_data(current_asset_info["cg_id"], CONFIG.get('coingecko',{}).get('vs_currency', 'usd'), _cg_days_to_fetch_param, current_interval_code)

        # 2. FILTRAGGIO PER DISPLAY e VALIDAZIONE
        if st.session_state.ss_data_ohlcv_full is not None and not st.session_state.ss_data_ohlcv_full.empty:
            logger.info(f"Storico completo caricato. Shape: {st.session_state.ss_data_ohlcv_full.shape}")
            # ... (logica di filtraggio per ss_data_ohlcv_display come prima, usando _display_start_date_dt e _display_end_date_dt)
            # ... ATTENZIONE: questa parte è cruciale per allineare i dati intraday/daily ...
            df_to_filter = st.session_state.ss_data_ohlcv_full
            if not isinstance(df_to_filter.index, pd.DatetimeIndex): df_to_filter.index = pd.to_datetime(df_to_filter.index)

            if interval_is_intraday: 
                 st.session_state.ss_data_ohlcv_display = df_to_filter[
                    (df_to_filter.index >= pd.to_datetime(_display_start_date_dt)) & 
                    (df_to_filter.index < pd.to_datetime(_display_end_date_dt) + pd.Timedelta(days=1)) 
                ].copy()
            else: 
                st.session_state.ss_data_ohlcv_display = df_to_filter[
                    (df_to_filter.index.normalize() >= pd.to_datetime(_display_start_date_dt).normalize()) & 
                    (df_to_filter.index.normalize() <= pd.to_datetime(_display_end_date_dt).normalize())
                ].copy()
            
            if st.session_state.ss_data_ohlcv_display.empty:
                 st.warning(f"Nessun dato per display dopo filtraggio.") # ...
            else:
                 st.success(f"Dati per display pronti. Shape: {st.session_state.ss_data_ohlcv_display.shape}")
        # ... (resto della pipeline come prima: feature su _full, training su _full, predizioni su _full, segnali su _full, poi filtra segnali per _display)

        # --- Esempio abbreviato della continuazione della pipeline ---
        if st.session_state.ss_data_ohlcv_full is not None and not st.session_state.ss_data_ohlcv_full.empty:
            progress_bar.progress(25, text="Calcolo feature...")
            st.session_state.ss_features_full = calculate_technical_features(st.session_state.ss_data_ohlcv_full)

            if st.session_state.ss_features_full.empty or len(st.session_state.ss_features_full) < 10: # O una soglia più adatta
                st.error("Fallimento calcolo feature o dati post-feature insufficienti per procedere.")
                logger.error("Fallimento calcolo feature o dati post-feature insuff.")
            else: # Prosegui solo se le feature sono state calcolate
                # ... (creazione target, training, predizioni, segnali SU DATI FULL) ...
                pred_horizon_periods = CONFIG.get('ml_model', {}).get('prediction_target_horizon_days', 3) # Questo potrebbe dover essere in "periodi" se intraday
                df_with_target_full = create_prediction_targets(st.session_state.ss_features_full, horizon=pred_horizon_periods)
                target_col_name = f'target_{pred_horizon_periods}d_pct_change' # o _periods_
                
                feature_cols_ml_config = CONFIG.get('ml_model',{}).get('feature_columns_for_training', ['MA20', 'MA50', 'RSI']) # Lista esempio
                feature_cols_for_ml = [col for col in feature_cols_ml_config if col in df_with_target_full.columns]

                if feature_cols_for_ml and target_col_name in df_with_target_full.columns:
                    st.session_state.ss_trained_ml_model = train_random_forest_model(df_with_target_full, feature_cols_for_ml, target_col_name)
                    if st.session_state.ss_trained_ml_model:
                        predictions_series = generate_model_predictions(st.session_state.ss_trained_ml_model, df_with_target_full, feature_cols_for_ml)
                        if predictions_series is not None:
                            st.session_state.ss_target_and_preds_full = df_with_target_full.copy()
                            st.session_state.ss_target_and_preds_full[f'prediction_{pred_horizon_periods}d_pct_change'] = predictions_series
                            
                            df_ml_signals_full = generate_signals_from_ml_predictions(st.session_state.ss_target_and_preds_full, f'prediction_{pred_horizon_periods}d_pct_change', CONFIG.get('signal_logic',{}).get('buy_threshold_change', 0.005), CONFIG.get('signal_logic',{}).get('sell_threshold_change', -0.005))
                            df_breakout_full = detect_breakout_signals(st.session_state.ss_features_full)
                            df_signals_combined_full = combine_signals(df_ml_signals_full, df_breakout_full)
                            df_signals_combined_full = apply_trading_spreads(df_signals_combined_full, current_asset_info["type"], CONFIG.get('spreads',{}))

                            # Aggiorna la tabella con gli ultimi segnali e prezzo per l'asset corrente
                            if not df_signals_combined_full.empty:
                                last_full_signal_row = df_signals_combined_full.iloc[-1]
                                st.session_state.ss_asset_table_data[current_asset_info["symbol"]]["ml_signal"] = last_full_signal_row.get('ml_signal', 'N/A')
                                st.session_state.ss_asset_table_data[current_asset_info["symbol"]]["breakout_signal"] = last_full_signal_row.get('breakout_signal', 'N/A')
                                st.session_state.ss_asset_table_data[current_asset_info["symbol"]]["last_price"] = f"{last_full_signal_row.get('Close', 0.0):.2f}" if 'Close' in last_full_signal_row else "N/A"
                            
                            # Filtra i segnali per l'intervallo di display
                            if st.session_state.ss_data_ohlcv_display is not None and not st.session_state.ss_data_ohlcv_display.empty:
                                common_idx_disp = st.session_state.ss_data_ohlcv_display.index.intersection(df_signals_combined_full.index)
                                if not common_idx_disp.empty:
                                    st.session_state.ss_final_signals_display = df_signals_combined_full.loc[common_idx_disp].copy()
                                    st.success(f"Segnali filtrati per display. Shape: {st.session_state.ss_final_signals_display.shape}")
                                    if not st.session_state.ss_final_signals_display.empty:
                                        last_disp_sig_row = st.session_state.ss_final_signals_display.iloc[-1]
                                        st.session_state.ss_last_signal_info_display = { # ... come prima }
                                            "ticker": current_asset_info["symbol"],
                                            "date": last_disp_sig_row.name.strftime('%Y-%m-%d %H:%M:%S') if isinstance(last_disp_sig_row.name, pd.Timestamp) else str(last_disp_sig_row.name),
                                            "ml_signal": last_disp_sig_row.get('ml_signal', 'N/A'),
                                            "breakout_signal": last_disp_sig_row.get('breakout_signal', 'N/A'),
                                            "close_price": f"{last_disp_sig_row.get('Close', 0.0):.2f}" if 'Close' in last_disp_sig_row else "N/A"
                                        }
                                        # Suoni/Email qui se necessario
                # ... (gestione errori e casi mancanti)
        else:
            st.error(f"Elaborazione ML non possibile: storico dati per {current_asset_info['symbol']} non caricato o vuoto.")
            logger.error(f"Pipeline ML saltata per {current_asset_info['symbol']} causa dati mancanti.")

        progress_bar.progress(100, text="Analisi completata!")
        time.sleep(0.5) 
        progress_bar.empty() 

    if st.session_state.get('ss_analysis_run_flag', False): 
        st.session_state.ss_analysis_run_flag = False
        logger.debug("Flag ss_analysis_run_flag resettato.")

# --- AREA PRINCIPALE PER VISUALIZZAZIONE RISULTATI ---
st.markdown("---")
st.header(f" деталі для: {st.session_state.ss_current_asset_display_info['name']} ({st.session_state.ss_current_asset_display_info['symbol']}) - {st.session_state.ss_current_asset_display_info['interval_label_short']}")

if st.session_state.ss_final_signals_display is not None and not st.session_state.ss_final_signals_display.empty:
    if st.session_state.ss_last_signal_info_display:
        # ... (Visualizzazione ultimo segnale)
        st.subheader("📢 Ultimo Segnale Generato (nell'intervallo visualizzato):")
        sig_info = st.session_state.ss_last_signal_info_display
        ml_color = "green" if sig_info['ml_signal'] == "BUY" else "red" if sig_info['ml_signal'] == "SELL" else "gray"
        breakout_color = "blue" if sig_info['breakout_signal'] == "BULLISH" else "orange" if sig_info['breakout_signal'] == "BEARISH" else "gray"
        st.markdown(f"""
        *   **Data Segnale:** `{sig_info['date']}`
        *   **Segnale ML:** <span style='color:{ml_color}; font-weight:bold;'>{sig_info['ml_signal']}</span>
        *   **Segnale Breakout:** <span style='color:{breakout_color};'>{sig_info['breakout_signal']}</span>
        *   **Prezzo Chiusura (al segnale):** `{sig_info['close_price']}`
        """, unsafe_allow_html=True)
        st.markdown("---")

    st.subheader("📈 Grafico Interattivo")
    df_features_for_chart = pd.DataFrame()
    # Per il grafico, usiamo i dati delle feature ma filtrati per l'intervallo di visualizzazione
    if st.session_state.ss_features_full is not None and not st.session_state.ss_features_full.empty and \
       st.session_state.ss_data_ohlcv_display is not None and not st.session_state.ss_data_ohlcv_display.empty:
        common_idx_chart = st.session_state.ss_data_ohlcv_display.index.intersection(st.session_state.ss_features_full.index)
        if not common_idx_chart.empty:
            df_features_for_chart = st.session_state.ss_features_full.loc[common_idx_chart].copy()
    
    if not df_features_for_chart.empty:
        chart_fig = create_main_stock_chart(
            df_ohlcv_ma=df_features_for_chart, 
            df_signals=st.session_state.ss_final_signals_display, 
            ticker=st.session_state.ss_current_asset_display_info["symbol"],
            ma_periods_to_show=CONFIG.get('visualization',{}).get('show_ma_periods', [20, 50])
        )
        st.plotly_chart(chart_fig, use_container_width=True)
    else:
        if st.session_state.get('ss_data_ohlcv_full') is not None: # Solo se l'analisi è stata tentata
             st.warning("Dati insufficienti o non allineati per visualizzare il grafico.")

    # ... (Expander con tabelle come prima, usando i DataFrame _display o _full filtrati per display)

elif 'ss_analysis_run_flag' in st.session_state and not st.session_state.ss_analysis_run_flag and st.session_state.get('ss_current_asset_display_info',{}).get('symbol'):
    # Questo blocco viene raggiunto se l'analisi non è in corso (o è finita) ma un asset è selezionato.
    # Potrebbe essere il caso dopo un errore nella pipeline o al primo caricamento.
    st.info(f"Pronto per analizzare {st.session_state.ss_current_asset_display_info['name']}. Clicca un intervallo nella tabella sopra.")
else: 
    st.info("👋 Benvenuto! Seleziona un asset e un intervallo dalla tabella per iniziare l'analisi.")

st.markdown("---")
# ... (Footer e Debug Session State come prima) ...
st.caption(f"Dashboard v{APP_VERSION_FROM_CONFIG}. Controlla i log del server per dettagli DEBUG/INFO.")
st.caption(f"Ultimo aggiornamento dell'interfaccia Streamlit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}")

with st.expander("🔍 DEBUG: Stato Sessione Completo", expanded=False): 
    session_state_dict_for_json = {}
    for k, v_item in st.session_state.to_dict().items(): 
        if isinstance(v_item, pd.DataFrame):
            session_state_dict_for_json[k] = f"DataFrame with shape {v_item.shape}" if v_item is not None else "None"
        elif isinstance(v_item, (datetime, date, pd.Timestamp, pd.Period)): 
             session_state_dict_for_json[k] = str(v_item)
        else:
            try: 
                json.dumps(v_item) 
                session_state_dict_for_json[k] = v_item
            except (TypeError, OverflowError): 
                session_state_dict_for_json[k] = str(v_item) 
    st.json(session_state_dict_for_json, expanded=False)
