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
CONFIG_FILE = "config.yaml"
CONFIG = {}
APP_VERSION_FROM_CONFIG = "N/A"

try:
    with open(CONFIG_FILE, 'r') as f:
        CONFIG = yaml.safe_load(f)
    APP_VERSION_FROM_CONFIG = CONFIG.get('version', 'v1.6.5-spec-impl (config fallback)')
    st.session_state.config_loaded_successfully = True
    st.write(f"DEBUG [app.py]: {CONFIG_FILE} caricato correttamente. Versione da config: {APP_VERSION_FROM_CONFIG}")
except FileNotFoundError:
    st.error(f"ERRORE CRITICO: Il file di configurazione '{CONFIG_FILE}' non è stato trovato. L'applicazione non può continuare.")
    st.session_state.config_loaded_successfully = False
    st.stop() # Ferma l'esecuzione se il file config manca
except yaml.YAMLError as e:
    st.error(f"ERRORE CRITICO nel parsing di '{CONFIG_FILE}': {e}. L'applicazione non può continuare.")
    st.session_state.config_loaded_successfully = False
    st.stop() # Ferma l'esecuzione se il file config è corrotto

# --- CONFIGURAZIONE PAGINA STREAMLIT ---
st.set_page_config(
    layout="wide",
    page_title=f"Stock Signal Dashboard {APP_VERSION_FROM_CONFIG}",
    page_icon="📈" # Emoji per icona
)

# --- GESTIONE CHIAVI API (da st.secrets) ---
# Alpha Vantage
ALPHA_VANTAGE_API_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY")
if not ALPHA_VANTAGE_API_KEY and CONFIG.get('alpha_vantage'): # Solo se Alpha Vantage è previsto
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
    CONFIG.get('ml_model', {}).get('google_ai_studio_url_secret_name'): GOOGLE_AI_STUDIO_URL,
    CONFIG.get('ml_model', {}).get('google_ai_studio_token_secret_name'): GOOGLE_AI_STUDIO_TOKEN,
    CONFIG.get('email_notifications', {}).get('smtp_password_secret_name'): EMAIL_SMTP_PASSWORD
}


# --- STATO DELLA SESSIONE (SESSION STATE) ---
# Inizializza valori di default per lo stato della sessione se non esistono già.
# Questo aiuta a mantenere l'input dell'utente e i risultati tra le interazioni.
default_session_state_values = {
    'ss_ticker_input': "AAPL", # Ticker di default
    'ss_asset_type': "stock",  # "stock" o "crypto"
    'ss_start_date': datetime.now() - timedelta(days=CONFIG.get('ml_model', {}).get('training_days', 90) + 180), # Periodo più lungo per avere dati sufficienti
    'ss_end_date': datetime.now(),
    'ss_raw_data_df': None,           # DataFrame con i dati grezzi OHLCV
    'ss_features_df': None,         # DataFrame con le feature calcolate
    'ss_target_and_preds_df': None, # DataFrame con target e predizioni ML
    'ss_final_signals_df': None,    # DataFrame con i segnali finali (ML, breakout)
    'ss_trained_ml_model': None,      # Oggetto del modello ML allenato
    'ss_last_generated_signal_info': None, # Dizionario con info sull'ultimo segnale
    'ss_analysis_run_triggered': False # Flag per indicare se l'analisi è stata avviata
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

    # Selezione Tipo Asset
    st.session_state.ss_asset_type = st.radio(
        "Seleziona Tipo Asset:",
        options=["stock", "crypto"],
        index=["stock", "crypto"].index(st.session_state.ss_asset_type), # Mantieni selezione precedente
        horizontal=True,
        key="asset_type_radio" # Chiave univoca per il widget
    )

    # Input Ticker/Coin ID (cambia placeholder in base al tipo di asset)
    default_ticker_placeholder = "Es. AAPL, TSLA" if st.session_state.ss_asset_type == "stock" else "Es. bitcoin, ethereum"
    current_ticker_value = st.session_state.get('ss_ticker_input', "AAPL" if st.session_state.ss_asset_type == "stock" else "bitcoin")
    
    st.session_state.ss_ticker_input = st.text_input(
        f"Inserisci Simbolo Ticker ({'Azioni' if st.session_state.ss_asset_type == 'stock' else 'Crypto'}):",
        value=current_ticker_value,
        placeholder=default_ticker_placeholder,
        key="ticker_text_input"
    )
    # Normalizza input: uppercase per stock, lowercase per crypto
    if st.session_state.ss_asset_type == "stock":
        st.session_state.ss_ticker_input = st.session_state.ss_ticker_input.upper()
    else:
        st.session_state.ss_ticker_input = st.session_state.ss_ticker_input.lower()


    # Selezione Intervallo Date
    st.markdown("##### Intervallo Date per Analisi:")
    col_date1, col_date2 = st.columns(2)
    with col_date1:
        st.session_state.ss_start_date = st.date_input("Data Inizio", value=st.session_state.ss_start_date, key="start_date_input")
    with col_date2:
        st.session_state.ss_end_date = st.date_input("Data Fine", value=st.session_state.ss_end_date, key="end_date_input")

    # Validazione date
    if st.session_state.ss_start_date > st.session_state.ss_end_date:
        st.error("Errore: La Data di Inizio deve essere precedente o uguale alla Data di Fine.")
        st.stop() # Ferma l'esecuzione se le date non sono valide

    # Bottone per avviare l'analisi completa
    st.markdown("---") # Divisore
    if st.button("🚀 Analizza e Genera Segnali", type="primary", use_container_width=True, key="run_analysis_button"):
        if not st.session_state.ss_ticker_input:
            st.warning("Per favore, inserisci un simbolo Ticker o Coin ID valido.")
        else:
            st.session_state.ss_analysis_run_triggered = True # Imposta il flag
            # Resetta gli stati dei dati precedenti per una nuova analisi
            st.session_state.ss_raw_data_df = None
            st.session_state.ss_features_df = None
            st.session_state.ss_target_and_preds_df = None
            st.session_state.ss_final_signals_df = None
            st.session_state.ss_trained_ml_model = None
            st.session_state.ss_last_generated_signal_info = None
            st.info(f"Avvio analisi per {st.session_state.ss_ticker_input} ({st.session_state.ss_asset_type})...")
    st.markdown("---")


# --- PIPELINE DI ELABORAZIONE DATI E SEGNALI (eseguita solo se il bottone è stato premuto) ---
if st.session_state.ss_analysis_run_triggered:
    st.markdown("### ⚙️ Log di Processo dell'Analisi")
    progress_bar = st.progress(0, text="Inizio analisi...")

    # 1. CARICAMENTO DATI GREZZI (OHLCV)
    progress_bar.progress(10, text=f"Caricamento dati per {st.session_state.ss_ticker_input}...")
    if st.session_state.ss_asset_type == "stock":
        if not ALPHA_VANTAGE_API_KEY:
            st.error("Impossibile caricare dati azioni: Chiave API Alpha Vantage mancante.")
        else:
            st.session_state.ss_raw_data_df = get_stock_data(
                ALPHA_VANTAGE_API_KEY, 
                st.session_state.ss_ticker_input,
                CONFIG.get('alpha_vantage', {}).get('function', 'TIME_SERIES_DAILY_ADJUSTED'),
                CONFIG.get('alpha_vantage', {}).get('outputsize', 'full') # 'full' per avere più storico
            )
    elif st.session_state.ss_asset_type == "crypto":
        days_history = (st.session_state.ss_end_date - st.session_state.ss_start_date).days + CONFIG.get('ml_model', {}).get('training_days', 90) # Assicura abbastanza dati per training
        if days_history <= 0: days_history = CONFIG.get('coingecko',{}).get('days_history', 90) + CONFIG.get('ml_model', {}).get('training_days', 90)
        
        st.session_state.ss_raw_data_df = get_crypto_data(
            st.session_state.ss_ticker_input,
            vs_currency=CONFIG.get('coingecko',{}).get('vs_currency', 'usd'),
            days=days_history # Carica storico sufficiente
        )
    
    # Filtra i dati per l'intervallo di date SELEZIONATO dall'utente DOPO il caricamento
    if st.session_state.ss_raw_data_df is not None and not st.session_state.ss_raw_data_df.empty:
        start_dt_filter = pd.to_datetime(st.session_state.ss_start_date)
        end_dt_filter = pd.to_datetime(st.session_state.ss_end_date)
        st.session_state.ss_raw_data_df = st.session_state.ss_raw_data_df[
            (st.session_state.ss_raw_data_df.index >= start_dt_filter) & 
            (st.session_state.ss_raw_data_df.index <= end_dt_filter)
        ]
        if st.session_state.ss_raw_data_df.empty:
            st.warning(f"Nessun dato per {st.session_state.ss_ticker_input} trovato nell'intervallo di date specificato ({st.session_state.ss_start_date.strftime('%Y-%m-%d')} - {st.session_state.ss_end_date.strftime('%Y-%m-%d')}).")
            st.session_state.ss_raw_data_df = None # Invalida se vuoto dopo filtro
        else:
            st.success(f"Dati grezzi per {st.session_state.ss_ticker_input} caricati e filtrati. Shape: {st.session_state.ss_raw_data_df.shape}")
    elif st.session_state.ss_raw_data_df is None: # Errore durante il caricamento
        st.error(f"Fallimento nel caricamento dei dati grezzi per {st.session_state.ss_ticker_input}.")
    
    # Continua solo se i dati grezzi sono stati caricati con successo
    if st.session_state.ss_raw_data_df is not None and not st.session_state.ss_raw_data_df.empty:
        progress_bar.progress(25, text="Calcolo feature tecniche...")
        # 2. CALCOLO FEATURE TECNICHE
        st.session_state.ss_features_df = calculate_technical_features(st.session_state.ss_raw_data_df)
        if st.session_state.ss_features_df.empty:
            st.error("Fallimento nel calcolo delle feature tecniche.")
        else:
            st.success(f"Feature tecniche calcolate. Shape: {st.session_state.ss_features_df.shape}")

            progress_bar.progress(40, text="Creazione target di predizione...")
            # 3. CREAZIONE TARGET PER PREDIZIONE ML
            pred_horizon = CONFIG.get('ml_model', {}).get('prediction_target_horizon_days', 3)
            df_with_target = create_prediction_targets(st.session_state.ss_features_df, horizon=pred_horizon)
            target_col_name = f'target_{pred_horizon}d_pct_change'

            # 4. TRAINING MODELLO ML E PREDIZIONI
            feature_cols_for_ml = CONFIG.get('ml_model',{}).get('feature_columns_for_training', ['MA20', 'MA50', 'RSI', 'StochRSI', 'Momentum'])
            # Rimuovi feature non esistenti nel DataFrame
            feature_cols_for_ml = [col for col in feature_cols_for_ml if col in df_with_target.columns]

            if not feature_cols_for_ml:
                st.error("Nessuna colonna feature valida trovata per il training/predizione ML.")
            elif target_col_name not in df_with_target.columns:
                st.error(f"Colonna target '{target_col_name}' non creata o mancante per il training ML.")
            else:
                if CONFIG.get('ml_model', {}).get('use_google_ai_studio', False):
                    progress_bar.progress(55, text="Ottenimento predizioni da Google AI Studio...")
                    # TODO: Implementare correttamente il passaggio dati e la gestione della risposta da AI Studio
                    # predictions_series = get_predictions_from_ai_studio(df_with_target[feature_cols_for_ml], CONFIG) 
                    st.warning("Integrazione Google AI Studio non completamente implementata in questo flusso.")
                    predictions_series = pd.Series(index=df_with_target.index, dtype=float) # Placeholder vuoto
                else: # Usa RandomForest locale
                    progress_bar.progress(55, text="Training modello RandomForest locale...")
                    st.session_state.ss_trained_ml_model = train_random_forest_model(
                        df_with_target, # Contiene già feature e target
                        feature_columns=feature_cols_for_ml,
                        target_column=target_col_name,
                        n_estimators=CONFIG.get('ml_model',{}).get('random_forest_n_estimators', 100)
                    )
                    if st.session_state.ss_trained_ml_model:
                        st.success("Modello RandomForest addestrato.")
                        progress_bar.progress(70, text="Generazione predizioni ML...")
                        predictions_series = generate_model_predictions(
                            st.session_state.ss_trained_ml_model,
                            df_with_target, # Passa il df con le feature
                            feature_columns=feature_cols_for_ml
                        )
                    else:
                        st.error("Fallimento training modello RandomForest.")
                        predictions_series = None

                if predictions_series is not None:
                    st.session_state.ss_target_and_preds_df = df_with_target.copy()
                    # Nome colonna predizione basato sull'orizzonte per chiarezza e coerenza
                    prediction_col_ml_name = f'prediction_{pred_horizon}d_pct_change'
                    st.session_state.ss_target_and_preds_df[prediction_col_ml_name] = predictions_series
                    st.success(f"Predizioni ML generate e aggiunte come '{prediction_col_ml_name}'.")
                else:
                    st.error("Fallimento nella generazione delle predizioni ML.")

            # 5. GENERAZIONE SEGNALI DI TRADING
            if st.session_state.ss_target_and_preds_df is not None and prediction_col_ml_name in st.session_state.ss_target_and_preds_df:
                progress_bar.progress(85, text="Generazione segnali di trading...")
                # Segnali dal modello ML
                df_ml_signals_only = generate_signals_from_ml_predictions(
                    st.session_state.ss_target_and_preds_df,
                    prediction_column_name=prediction_col_ml_name,
                    buy_threshold=CONFIG.get('signal_logic',{}).get('buy_threshold_change', 0.005),
                    sell_threshold=CONFIG.get('signal_logic',{}).get('sell_threshold_change', -0.005)
                )
                # Segnali di Breakout (usano ss_features_df che ha OHLCV e MA)
                df_breakout_signals_only = detect_breakout_signals(
                    st.session_state.ss_features_df, # Assicurati che abbia le colonne OHLCV
                    high_low_period=CONFIG.get('signal_logic',{}).get('breakout_days_high_low_period', 20),
                    volume_avg_factor=CONFIG.get('signal_logic',{}).get('breakout_volume_avg_factor', 1.0),
                    volume_period=CONFIG.get('signal_logic',{}).get('breakout_days_high_low_period', 20) # Usa lo stesso periodo per volume medio
                )
                # Combina segnali ML e Breakout
                # Il DataFrame df_ml_signals_only contiene già molte colonne da ss_target_and_preds_df
                # Quindi, passiamolo come base e uniamo solo la colonna 'breakout_signal'.
                st.session_state.ss_final_signals_df = combine_signals(df_ml_signals_only, df_breakout_signals_only)

                # Applica spread (logica placeholder per ora)
                st.session_state.ss_final_signals_df = apply_trading_spreads(
                    st.session_state.ss_final_signals_df,
                    st.session_state.ss_asset_type,
                    CONFIG.get('spreads', {})
                )
                st.success("Segnali di trading finali generati.")

                # Gestisci ultimo segnale per notifiche/audio
                if not st.session_state.ss_final_signals_df.empty:
                    last_signal_row = st.session_state.ss_final_signals_df.iloc[-1]
                    st.session_state.ss_last_generated_signal_info = {
                        "ticker": st.session_state.ss_ticker_input,
                        "date": last_signal_row.name.strftime('%Y-%m-%d %H:%M:%S') if isinstance(last_signal_row.name, pd.Timestamp) else str(last_signal_row.name),
                        "ml_signal": last_signal_row.get('ml_signal', 'N/A'),
                        "breakout_signal": last_signal_row.get('breakout_signal', 'N/A'),
                        "close_price": f"{last_signal_row.get('Close', 0.0):.2f}" if 'Close' in last_signal_row else "N/A"
                    }
                    # Suoni e notifiche email
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
        
        progress_bar.progress(100, text="Analisi completata!")
        time.sleep(1) # Lascia la barra al 100% per un secondo
        progress_bar.empty() # Rimuovi la barra di progresso

    st.session_state.ss_analysis_run_triggered = False # Resetta il flag per evitare riesecuzione automatica al prossimo re-run di Streamlit


# --- AREA PRINCIPALE PER VISUALIZZAZIONE RISULTATI ---
st.markdown("---")
st.header(f"📊 Risultati per: {st.session_state.ss_ticker_input if st.session_state.ss_ticker_input else 'N/D'}")

if st.session_state.ss_final_signals_df is not None and not st.session_state.ss_final_signals_df.empty:
    # Visualizza l'ultimo segnale generato in modo prominente
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

    # Grafico Interattivo
    st.subheader("📈 Grafico Interattivo con Segnali")
    # Il DataFrame per il grafico OHLCV e MA è ss_features_df (contiene OHLCV + MA)
    # Il DataFrame per i segnali è ss_final_signals_df (contiene ml_signal, breakout_signal)
    chart_fig = create_main_stock_chart(
        df_ohlcv_ma=st.session_state.ss_features_df, 
        df_signals=st.session_state.ss_final_signals_df,
        ticker=st.session_state.ss_ticker_input,
        ma_periods_to_show=CONFIG.get('visualization',{}).get('show_ma_periods', [20, 50])
    )
    st.plotly_chart(chart_fig, use_container_width=True)

    # Espansori per mostrare i DataFrame dettagliati
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

elif st.session_state.get('ss_ticker_input'): # Se è stato inserito un ticker ma non ci sono risultati (es. dopo un errore)
    st.info("Nessun risultato da visualizzare. Verifica i parametri di input e i log di processo sopra, poi clicca 'Analizza e Genera Segnali'.")
else: # All'avvio, prima di qualsiasi input
    st.info("👋 Benvenuto! Inserisci i parametri nella sidebar a sinistra e clicca 'Analizza e Genera Segnali' per iniziare.")

# Footer o informazioni aggiuntive
st.markdown("---")
st.caption(f"Dashboard v{APP_VERSION_FROM_CONFIG}. Controlla i messaggi DEBUG e gli errori nel log di processo per dettagli sull'esecuzione.")
st.caption(f"Ultimo aggiornamento dell'interfaccia Streamlit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}")

# Per debug: mostra lo stato della sessione completo nella sidebar
with st.sidebar.expander("🔍 DEBUG: Stato Sessione Completo", expanded=False):
    st.json(st.session_state.to_dict(), expanded=False)
