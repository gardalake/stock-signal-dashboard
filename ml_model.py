# ml_model.py - v1.6.5 (Technical features with 'ta' library - Python Logger)
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import streamlit as st # Mantenuto per st.error/st.warning diretti all'utente

# Importa il setup del logger
from logger_utils import setup_logger
logger = setup_logger(__name__) # Configura un logger per questo modulo

# Importa la libreria 'ta' (Technical Analysis library)
TA_AVAILABLE = False 
try:
    import ta
    TA_AVAILABLE = True
    logger.info("Libreria 'ta' importata con successo.")
except ImportError:
    logger.warning("Libreria 'ta' non trovata. Feature tecniche limitate. Installala con 'pip install ta'.")


def calculate_technical_features(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola le feature tecniche usando la libreria 'ta'.
    """
    logger.debug("Inizio calcolo feature tecniche con libreria 'ta'.")
    df = df_input.copy()

    if not TA_AVAILABLE:
        st.error("[ML_MODEL] ERRORE: Libreria 'ta' non disponibile. Impossibile calcolare le feature tecniche complete.")
        logger.error("Libreria 'ta' non disponibile, calcolo feature interrotto/limitato.")
        if 'Close' in df.columns: # Fallback a feature molto semplici
            df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
            df['MA50'] = df['Close'].rolling(window=50, min_periods=1).mean()
            df['RSI'] = 50.0 
            df['StochRSI_K'] = 0.5 
            df['StochRSI_D'] = 0.5
            df['Momentum_ROC10'] = df['Close'].pct_change(periods=10) * 100 
        return df.dropna() 

    if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
        st.error("[ML_MODEL] ERRORE: Colonne OHLCV necessarie non trovate per il calcolo delle feature con 'ta'.")
        logger.error("Colonne OHLCV mancanti per calcolo feature con 'ta'.")
        return pd.DataFrame() 

    if df[['Open', 'High', 'Low', 'Close', 'Volume']].isnull().any().any():
        st.warning("[ML_MODEL] ATTENZIONE: Trovati NaN nei dati OHLCV. Questo potrebbe influenzare il calcolo degli indicatori.")
        logger.warning("Trovati NaN nei dati OHLCV che potrebbero influenzare il calcolo degli indicatori 'ta'.")
        
    try:
        # df['Volume_SMA20'] = ta.volume.VolumeWeightedAveragePrice(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=20, fillna=True).vwap
        df['Volume_SMA20'] = ta.trend.sma_indicator(close=df['Volume'], window=20, fillna=True)

        bollinger = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2, fillna=True)
        df['Bollinger_High'] = bollinger.bollinger_hband()
        df['Bollinger_Low'] = bollinger.bollinger_lband()
        df['Bollinger_Mid'] = bollinger.bollinger_mavg() 

        df['MA20'] = ta.trend.sma_indicator(close=df['Close'], window=20, fillna=True)
        df['MA50'] = ta.trend.sma_indicator(close=df['Close'], window=50, fillna=True)
        adx_indicator = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14, fillna=True) 
        df['ADX'] = adx_indicator.adx()
        df['ADX_Pos'] = adx_indicator.adx_pos()
        df['ADX_Neg'] = adx_indicator.adx_neg()

        df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14, fillna=True).rsi()
        
        stoch_rsi_indicator = ta.momentum.StochRSIIndicator(close=df['Close'], window=14, smooth1=3, smooth2=3, fillna=True)
        df['StochRSI_K'] = stoch_rsi_indicator.stochrsi_k()
        df['StochRSI_D'] = stoch_rsi_indicator.stochrsi_d()
        
        df['WilliamsR'] = ta.momentum.WilliamsRIndicator(high=df['High'], low=df['Low'], close=df['Close'], lbp=14, fillna=True).williams_r()
        
        df['Momentum_ROC10'] = ta.momentum.ROCIndicator(close=df['Close'], window=10, fillna=True).roc()
        
        macd_indicator = ta.trend.MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9, fillna=True)
        df['MACD_line'] = macd_indicator.macd()
        df['MACD_signal'] = macd_indicator.macd_signal()
        df['MACD_hist'] = macd_indicator.macd_diff() 

        logger.debug(f"Feature tecniche (con 'ta') calcolate. Shape: {df.shape}. Colonne: {df.columns.tolist()}")
        
    except Exception as e:
        st.error(f"[ML_MODEL] ERRORE durante il calcolo delle feature con 'ta': {e}")
        logger.error(f"Errore calcolo feature con 'ta': {e}", exc_info=True)
        return df_input 

    return df


def create_prediction_targets(df_input: pd.DataFrame, horizon: int) -> pd.DataFrame:
    logger.debug(f"Creazione target di predizione per orizzonte {horizon} giorni.")
    df = df_input.copy()
    target_col_name = f'target_{horizon}d_pct_change'
    
    if 'Close' not in df.columns:
        st.error(f"[ML_MODEL] ERRORE: Colonna 'Close' non trovata per creare il target '{target_col_name}'.")
        logger.error(f"Colonna 'Close' non trovata per creare target '{target_col_name}'.")
        return df 

    df[target_col_name] = df['Close'].shift(-horizon) / df['Close'] - 1.0
    
    logger.debug(f"Colonna target '{target_col_name}' creata.")
    return df


def train_random_forest_model(
    df_features_and_target: pd.DataFrame, 
    feature_columns: list, 
    target_column: str, 
    n_estimators: int = 100,
    random_state: int = 42
) -> RandomForestRegressor | None:
    logger.info(f"Inizio training RandomForest per target '{target_column}'. Feature: {feature_columns}")
    
    valid_feature_columns = [col for col in feature_columns if col in df_features_and_target.columns]
    if not valid_feature_columns:
        st.error(f"[ML_MODEL] ERRORE: Nessuna feature specificata ({feature_columns}) trovata nel DataFrame per il training.")
        logger.error(f"Nessuna feature valida trovata per il training: {feature_columns}")
        return None
    if len(valid_feature_columns) < len(feature_columns):
        missing_cols = set(feature_columns) - set(valid_feature_columns)
        st.warning(f"[ML_MODEL] ATTENZIONE: Alcune feature specificate non trovate: {missing_cols}. Uso solo: {valid_feature_columns}")
        logger.warning(f"Feature mancanti per training: {missing_cols}. Uso solo: {valid_feature_columns}")
    
    if target_column not in df_features_and_target.columns:
        st.error(f"[ML_MODEL] ERRORE: Target column '{target_column}' non trovata nel DataFrame prima di dropna.")
        logger.error(f"Target column '{target_column}' non trovata.")
        return None
        
    df_train = df_features_and_target.dropna(subset=valid_feature_columns + [target_column]) 

    if df_train.empty:
        st.error("[ML_MODEL] ERRORE: Nessun dato valido per il training dopo la rimozione dei NaN (feature o target).")
        logger.error("Nessun dato valido per il training dopo dropna.")
        return None
    
    X_train = df_train[valid_feature_columns]
    y_train = df_train[target_column]

    if X_train.empty or len(X_train) < 10: 
        st.error(f"[ML_MODEL] ERRORE: Dati di training insufficienti (campioni: {len(X_train)}). Minimo 10 richiesti.")
        logger.error(f"Dati di training insufficienti: {len(X_train)} campioni.")
        return None
        
    model = RandomForestRegressor(
        n_estimators=n_estimators, 
        random_state=random_state, 
        n_jobs=-1, 
        max_depth=10, 
        min_samples_split=10, 
        min_samples_leaf=5   
    )
    
    try:
        model.fit(X_train, y_train)
        logger.info(f"Modello RandomForest per '{target_column}' addestrato con successo.")
        return model
    except Exception as e:
        st.error(f"[ML_MODEL] ERRORE durante il training del RandomForest: {e}")
        logger.error(f"Errore training RandomForest: {e}", exc_info=True)
        return None


def generate_model_predictions(
    model: RandomForestRegressor, 
    df_with_features: pd.DataFrame, 
    feature_columns: list
) -> pd.Series | None:
    logger.debug(f"Inizio generazione predizioni con feature: {feature_columns}")
    
    if df_with_features.empty:
        st.error("[ML_MODEL] ERRORE: DataFrame vuoto fornito per la predizione.")
        logger.error("DataFrame vuoto per la predizione.")
        return None

    valid_feature_columns = [col for col in feature_columns if col in df_with_features.columns]
    if not valid_feature_columns:
        st.error(f"[ML_MODEL] ERRORE: Nessuna feature specificata ({feature_columns}) trovata per la predizione.")
        logger.error(f"Nessuna feature valida per la predizione: {feature_columns}")
        return None
    if len(valid_feature_columns) < len(feature_columns):
        missing_cols = set(feature_columns) - set(valid_feature_columns)
        st.warning(f"[ML_MODEL] ATTENZIONE: Alcune feature specificate non trovate per predizione: {missing_cols}. Uso solo: {valid_feature_columns}")
        logger.warning(f"Feature mancanti per predizione: {missing_cols}. Uso solo: {valid_feature_columns}")

    predictions_aligned = pd.Series(index=df_with_features.index, dtype=float, name="prediction")
    idx_for_prediction = df_with_features[valid_feature_columns].dropna().index
    
    if idx_for_prediction.empty:
        st.warning("[ML_MODEL] ATTENZIONE: Nessun dato con feature valide (non-NaN) per la predizione.")
        logger.warning("Nessun dato con feature valide (non-NaN) per la predizione.")
        return predictions_aligned 

    X_predict = df_with_features.loc[idx_for_prediction, valid_feature_columns]
    
    try:
        predictions_subset = model.predict(X_predict)
        predictions_aligned.loc[idx_for_prediction] = predictions_subset 
        
        logger.debug(f"Predizioni generate. Valide: {len(predictions_subset)}. Totale: {len(predictions_aligned)}")
        return predictions_aligned
    except Exception as e:
        st.error(f"[ML_MODEL] ERRORE durante la generazione delle predizioni: {e}")
        logger.error(f"Errore generazione predizioni: {e}", exc_info=True)
        return None


def get_predictions_from_ai_studio(df_features: pd.DataFrame, config: dict) -> pd.Series | None:
    st.info("[ML_MODEL] Integrazione Google AI Studio non ancora implementata.") 
    logger.info("Chiamata a get_predictions_from_ai_studio (non implementato).")
    return pd.Series(index=df_features.index, dtype=float, name="prediction_ai_studio") 


if __name__ == '__main__':
    if 'logger' not in locals(): # Fallback se il logger non è stato inizializzato sopra
        import logging
        logger = logging.getLogger(__name__)
        if not logger.handlers: # Evita handler duplicati
            handler = logging.StreamHandler(sys.stdout) # sys non è importato qui, sarebbe necessario
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)

    logger.info("--- INIZIO TEST STANDALONE ml_model.py ---")
    if not TA_AVAILABLE:
        logger.error("Libreria 'ta' non disponibile, test limitato.")
    else:
        logger.info("Libreria 'ta' disponibile per il test.")

    sample_data_ohlcv = {
        'Open':  [i for i in range(100, 160, 2)],
        'High':  [i + 1 for i in range(100, 160, 2)],
        'Low':   [i - 1 for i in range(100, 160, 2)],
        'Close': [i + 0.5 for i in range(100, 160, 2)],
        'Volume':[j * 1000 for j in range(10, 40)]
    }
    num_periods = len(sample_data_ohlcv['Close'])
    sample_dates_ml = pd.date_range(start='2023-01-01', periods=num_periods, freq='B')
    df_sample_ml = pd.DataFrame(sample_data_ohlcv, index=sample_dates_ml)
    logger.info("DataFrame di Esempio Iniziale (OHLCV):\n" + df_sample_ml.head().to_string())

    df_with_features_ml = calculate_technical_features(df_sample_ml)
    logger.info("\nDataFrame con Feature Tecniche (da 'ta'):\n" + df_with_features_ml.tail().to_string())

    prediction_horizon_ml = 3 
    df_with_target_ml = create_prediction_targets(df_with_features_ml, horizon=prediction_horizon_ml)
    logger.info(f"\nDataFrame con Target di Predizione ({prediction_horizon_ml}d % change):\n" + df_with_target_ml.tail(prediction_horizon_ml + 5).to_string())

    feature_cols_ml_test = ['MA20', 'MA50', 'RSI', 'StochRSI_K', 'StochRSI_D', 'WilliamsR', 'Momentum_ROC10', 'ADX', 'MACD_line', 'MACD_signal', 'MACD_hist', 'Bollinger_High', 'Bollinger_Low', 'Volume_SMA20']
    feature_cols_ml_test = [col for col in feature_cols_ml_test if col in df_with_target_ml.columns]
    target_col_ml_test = f'target_{prediction_horizon_ml}d_pct_change'
    
    logger.info(f"\nTraining RandomForest con feature: {feature_cols_ml_test} e target: {target_col_ml_test}")
    
    df_for_training_test = df_with_target_ml.copy() 
    if target_col_ml_test in df_for_training_test.columns and feature_cols_ml_test:
        df_for_training_test.dropna(subset=feature_cols_ml_test + [target_col_ml_test], inplace=True)
    else:
        logger.error("Target o feature columns mancanti prima di dropna nel test.")
        df_for_training_test = pd.DataFrame() 

    if not df_for_training_test.empty and len(df_for_training_test) >=10 :
        trained_rf_model_ml = train_random_forest_model(
            df_for_training_test, 
            feature_columns=feature_cols_ml_test,
            target_column=target_col_ml_test,
            n_estimators=10 
        )

        if trained_rf_model_ml:
            logger.info("Modello RandomForest addestrato con successo (test).")
            predictions_series_ml = generate_model_predictions(
                trained_rf_model_ml,
                df_with_target_ml, 
                feature_columns=feature_cols_ml_test
            )
            if predictions_series_ml is not None:
                logger.info("\nSerie di Predizioni Generate:")
                df_final_test_ml = df_with_target_ml.copy()
                df_final_test_ml['prediction'] = predictions_series_ml
                logger.info("\n" + df_final_test_ml[['Close', target_col_ml_test, 'prediction'] + feature_cols_ml_test].tail(10).to_string())
            else:
                logger.error("Fallita generazione predizioni (test).")
        else:
            logger.error("Fallito training del modello RandomForest (test).")
    else:
        logger.warning(f"Dati insufficienti per il training dopo dropna. Righe: {len(df_for_training_test)}. Feature disponibili: {feature_cols_ml_test}")
        
    logger.info("\n--- FINE TEST STANDALONE ml_model.py ---")
