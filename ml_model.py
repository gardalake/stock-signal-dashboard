# ml_model.py - v1.6.5 (Technical features with 'ta' library)
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import streamlit as st # Per debug e messaggi

# Importa la libreria 'ta' (Technical Analysis library)
try:
    import ta
    TA_AVAILABLE = True
    st.write("DEBUG [ml_model]: Libreria 'ta' importata con successo.")
except ImportError:
    TA_AVAILABLE = False
    st.warning("ATTENZIONE [ml_model]: Libreria 'ta' non trovata. Le feature tecniche saranno limitate o placeholder. Installala con 'pip install ta'.")


def calculate_technical_features(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola le feature tecniche usando la libreria 'ta'.
    Feature: MA20, MA50, RSI, StochRSI_K, StochRSI_D, Momentum (Williams %R come proxy o diff).
    """
    st.write("DEBUG [ml_model]: Inizio calcolo feature tecniche con libreria 'ta'.")
    df = df_input.copy()

    if not TA_AVAILABLE:
        st.error("[ml_model] ERRORE: Libreria 'ta' non disponibile. Impossibile calcolare le feature tecniche complete.")
        # Fallback a feature molto semplici se 'ta' non c'è
        if 'Close' in df.columns:
            df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
            df['MA50'] = df['Close'].rolling(window=50, min_periods=1).mean()
            df['RSI'] = 50.0 
            df['StochRSI_K'] = 0.5 
            df['StochRSI_D'] = 0.5
            df['Momentum_ROC'] = df['Close'].pct_change(periods=10) * 100 # Rate of Change 10 periodi
        return df.dropna() # Rimuovi i NaN iniziali

    if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
        st.error("[ml_model] ERRORE: Colonne OHLCV necessarie non trovate per il calcolo delle feature con 'ta'.")
        # Potremmo voler restituire df o un df vuoto a seconda della gravità
        return pd.DataFrame() 

    # Pulisci eventuali righe con NaN in OHLCV prima di calcolare gli indicatori
    # Questo è importante perché 'ta' può fallire o dare risultati strani con NaN.
    # df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True) # Meglio farlo prima di chiamare questa funzione
    if df[['Open', 'High', 'Low', 'Close', 'Volume']].isnull().any().any():
        st.warning("[ml_model] ATTENZIONE: Trovati NaN nei dati OHLCV. Questo potrebbe influenzare il calcolo degli indicatori. Considera di pulire i dati prima.")
        # 'ta' generalmente gestisce i NaN iniziali (restituendo NaN), ma non i NaN sparsi nel mezzo.

    try:
        # Aggiungi tutte le feature usando 'ta'
        # Volume
        df['Volume_SMA20'] = ta.volume.VolumeWeightedAveragePrice(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=20).vwap

        # Volatility
        bollinger = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['Bollinger_High'] = bollinger.bollinger_hband()
        df['Bollinger_Low'] = bollinger.bollinger_lband()
        df['Bollinger_Mid'] = bollinger.bollinger_mavg() # È una SMA20

        # Trend
        df['MA20'] = ta.trend.sma_indicator(close=df['Close'], window=20)
        df['MA50'] = ta.trend.sma_indicator(close=df['Close'], window=50)
        # Esempio: ADX
        adx_indicator = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
        df['ADX'] = adx_indicator.adx()
        df['ADX_Pos'] = adx_indicator.adx_pos()
        df['ADX_Neg'] = adx_indicator.adx_neg()

        # Momentum
        df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
        
        stoch_rsi_indicator = ta.momentum.StochRSIIndicator(close=df['Close'], window=14, smooth1=3, smooth2=3)
        df['StochRSI_K'] = stoch_rsi_indicator.stochrsi_k()
        df['StochRSI_D'] = stoch_rsi_indicator.stochrsi_d()
        
        # Williams %R come indicatore di momentum
        df['WilliamsR'] = ta.momentum.WilliamsRIndicator(high=df['High'], low=df['Low'], close=df['Close'], lbp=14).williams_r()
        
        # Rate of Change (ROC) come altra misura di momentum
        df['Momentum_ROC10'] = ta.momentum.ROCIndicator(close=df['Close'], window=10).roc()
        
        # MACD
        macd_indicator = ta.trend.MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
        df['MACD_line'] = macd_indicator.macd()
        df['MACD_signal'] = macd_indicator.macd_signal()
        df['MACD_hist'] = macd_indicator.macd_diff() # Istogramma MACD

        st.write(f"DEBUG [ml_model]: Feature tecniche (con 'ta') calcolate. Shape: {df.shape}")
        
        # Rimuovi righe con NaN risultanti dal calcolo degli indicatori (specialmente all'inizio)
        # df.dropna(inplace=True) # Fare attenzione con dropna() qui, potrebbe rimuovere troppo.
        # È meglio che il chiamante gestisca i NaN prima del training.
        # Però, per le feature, è comune rimuovere i NaN iniziali.
        # Il numero di righe con NaN dipende dalla finestra più lunga usata (es. MA50).
        # Se il DataFrame è piccolo, questo potrebbe ridurlo a zero.
        # Una strategia potrebbe essere di rimuovere solo se TUTTE le feature sono NaN per una riga.
        
    except Exception as e:
        st.error(f"[ml_model] ERRORE durante il calcolo delle feature con 'ta': {e}")
        # Restituisci il DataFrame originale (o parzialmente processato) in caso di errore
        return df_input 

    return df


def create_prediction_targets(df_input: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Crea la colonna target per la predizione.
    Il target è la variazione percentuale del prezzo di chiusura 'horizon' giorni nel futuro.
    """
    st.write(f"DEBUG [ml_model]: Creazione target di predizione per orizzonte {horizon} giorni.")
    df = df_input.copy()
    target_col_name = f'target_{horizon}d_pct_change'
    
    if 'Close' not in df.columns:
        st.error(f"[ml_model] ERRORE: Colonna 'Close' non trovata per creare il target '{target_col_name}'.")
        return df 

    df[target_col_name] = df['Close'].shift(-horizon) / df['Close'] - 1.0
    
    st.write(f"DEBUG [ml_model]: Colonna target '{target_col_name}' creata.")
    return df


def train_random_forest_model(
    df_features_and_target: pd.DataFrame, 
    feature_columns: list, 
    target_column: str, 
    n_estimators: int = 100,
    random_state: int = 42
) -> RandomForestRegressor | None:
    st.write(f"DEBUG [ml_model]: Inizio training RandomForest per target '{target_column}'. Feature: {feature_columns}")
    
    # Assicurati che le feature_columns esistano effettivamente nel DataFrame
    valid_feature_columns = [col for col in feature_columns if col in df_features_and_target.columns]
    if not valid_feature_columns:
        st.error(f"[ml_model] ERRORE: Nessuna delle feature specificate ({feature_columns}) trovata nel DataFrame per il training.")
        return None
    if len(valid_feature_columns) < len(feature_columns):
        st.warning(f"[ml_model] ATTENZIONE: Alcune feature specificate non trovate. Uso solo: {valid_feature_columns}")
    
    df_train = df_features_and_target.dropna(subset=valid_feature_columns + [target_column]) 

    if df_train.empty:
        st.error("[ml_model] ERRORE: Nessun dato valido per il training dopo la rimozione dei NaN (feature o target).")
        return None
    
    if target_column not in df_train.columns:
        st.error(f"[ml_model] ERRORE: Target column '{target_column}' non trovata nel DataFrame di training.")
        return None

    X_train = df_train[valid_feature_columns]
    y_train = df_train[target_column]

    if X_train.empty or len(X_train) < 10: # Aggiunto controllo per un numero minimo di campioni
        st.error(f"[ml_model] ERRORE: Dati di training insufficienti (campioni: {len(X_train)}). Minimo 10 richiesti.")
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
        st.write(f"DEBUG [ml_model]: Modello RandomForest per '{target_column}' addestrato con successo.")
        return model
    except Exception as e:
        st.error(f"[ml_model] ERRORE durante il training del RandomForest: {e}")
        return None


def generate_model_predictions(
    model: RandomForestRegressor, 
    df_with_features: pd.DataFrame, 
    feature_columns: list
) -> pd.Series | None:
    st.write(f"DEBUG [ml_model]: Inizio generazione predizioni con feature: {feature_columns}")
    
    if df_with_features.empty:
        st.error("[ml_model] ERRORE: DataFrame vuoto fornito per la predizione.")
        return None

    valid_feature_columns = [col for col in feature_columns if col in df_with_features.columns]
    if not valid_feature_columns:
        st.error(f"[ml_model] ERRORE: Nessuna delle feature specificate ({feature_columns}) trovata nel DataFrame per la predizione.")
        return None
    if len(valid_feature_columns) < len(feature_columns):
        st.warning(f"[ml_model] ATTENZIONE: Alcune feature specificate non trovate per predizione. Uso solo: {valid_feature_columns}")

    df_predict_valid_features = df_with_features.copy()
    # Non droppare NaN qui per le predizioni, il modello è stato allenato su dati non-NaN.
    # Le predizioni verranno fatte dove le feature sono disponibili, altrimenti saranno NaN.
    # Se si droppano NaN qui, l'indice della predizione non corrisponderà più a df_with_features.

    # Crea una Series di NaN con l'indice corretto
    predictions_aligned = pd.Series(index=df_predict_valid_features.index, dtype=float, name="prediction")
    
    # Seleziona solo le righe dove TUTTE le feature valide sono non-NaN per la predizione
    idx_for_prediction = df_predict_valid_features[valid_feature_columns].dropna().index
    
    if idx_for_prediction.empty:
        st.warning("[ml_model] ATTENZIONE: Nessun dato con feature valide (non-NaN) per la predizione.")
        return predictions_aligned # Ritorna la series di NaN

    X_predict = df_predict_valid_features.loc[idx_for_prediction, valid_feature_columns]
    
    try:
        predictions_subset = model.predict(X_predict)
        predictions_aligned.loc[idx_for_prediction] = predictions_subset # Assegna le predizioni all'indice corretto
        
        st.write(f"DEBUG [ml_model]: Predizioni generate. Numero di predizioni valide: {len(predictions_subset)}. Lunghezza totale Series: {len(predictions_aligned)}")
        return predictions_aligned
    except Exception as e:
        st.error(f"[ml_model] ERRORE durante la generazione delle predizioni: {e}")
        return None


def get_predictions_from_ai_studio(df_features: pd.DataFrame, config: dict) -> pd.Series | None:
    st.info("[ml_model]: Integrazione Google AI Studio non ancora implementata.")
    return pd.Series(index=df_features.index, dtype=float, name="prediction_ai_studio") 


if __name__ == '__main__':
    st.write("--- INIZIO TEST STANDALONE ml_model.py ---")
    if not TA_AVAILABLE:
        st.error("Libreria 'ta' non disponibile, test limitato.")
    else:
        st.success("Libreria 'ta' disponibile per il test.")

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
    st.write("DataFrame di Esempio Iniziale (OHLCV):")
    st.dataframe(df_sample_ml.head())

    df_with_features_ml = calculate_technical_features(df_sample_ml)
    st.write("\nDataFrame con Feature Tecniche (da 'ta'):")
    st.dataframe(df_with_features_ml.tail()) 

    prediction_horizon_ml = 3 
    df_with_target_ml = create_prediction_targets(df_with_features_ml, horizon=prediction_horizon_ml)
    st.write(f"\nDataFrame con Target di Predizione ({prediction_horizon_ml}d % change):")
    st.dataframe(df_with_target_ml.tail(prediction_horizon_ml + 5))

    # Feature da usare per il modello, assicurati che siano quelle calcolate da 'ta'
    # Devono essere presenti in df_with_target_ml.columns
    feature_cols_ml_test = ['MA20', 'MA50', 'RSI', 'StochRSI_K', 'StochRSI_D', 'WilliamsR', 'Momentum_ROC10', 'ADX', 'MACD_line', 'MACD_signal', 'MACD_hist', 'Bollinger_High', 'Bollinger_Low']
    # Filtra solo quelle effettivamente presenti
    feature_cols_ml_test = [col for col in feature_cols_ml_test if col in df_with_target_ml.columns]

    target_col_ml_test = f'target_{prediction_horizon_ml}d_pct_change'
    
    st.write(f"\nTraining RandomForest con feature: {feature_cols_ml_test} e target: {target_col_ml_test}")
    
    # Rimuovi righe con NaN nelle feature e nel target prima del training
    df_for_training_test = df_with_target_ml.dropna(subset=feature_cols_ml_test + [target_col_ml_test])

    if not df_for_training_test.empty and len(df_for_training_test) >=10 :
        trained_rf_model_ml = train_random_forest_model(
            df_for_training_test, # Passa il DataFrame già pulito
            feature_columns=feature_cols_ml_test,
            target_column=target_col_ml_test,
            n_estimators=10 
        )

        if trained_rf_model_ml:
            st.success("Modello RandomForest addestrato con successo (test).")
            
            # Per la predizione, usa df_with_target_ml (che ha i NaN iniziali per le feature)
            # La funzione generate_model_predictions gestirà i NaN per le feature.
            predictions_series_ml = generate_model_predictions(
                trained_rf_model_ml,
                df_with_target_ml, 
                feature_columns=feature_cols_ml_test
            )
            if predictions_series_ml is not None:
                st.write("\nSerie di Predizioni Generate:")
                df_final_test_ml = df_with_target_ml.copy()
                df_final_test_ml['prediction'] = predictions_series_ml
                st.dataframe(df_final_test_ml[['Close', target_col_ml_test, 'prediction'] + feature_cols_ml_test].tail(10))
            else:
                st.error("Fallita generazione predizioni (test).")
        else:
            st.error("Fallito training del modello RandomForest (test).")
    else:
        st.warning(f"Dati insufficienti per il training dopo dropna. Righe: {len(df_for_training_test)}. Feature: {feature_cols_ml_test}")
        
    st.write("\n--- FINE TEST STANDALONE ml_model.py ---")
