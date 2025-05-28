# ml_model.py - v1.6.5
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
# import numpy as np # Non usato direttamente qui per ora
import streamlit as st # Per debug e messaggi

# Per gli indicatori tecnici, la libreria 'ta' è molto comoda.
# Se decidi di usarla, aggiungila a requirements.txt e decommenta l'import.
# import ta

def calculate_technical_features(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola le feature tecniche richieste: MA20, MA50, RSI, StochRSI, Momentum.
    Attualmente usa implementazioni placeholder o molto semplici.
    Considera l'uso della libreria 'ta' per implementazioni robuste.
    """
    st.write("DEBUG [ml_model]: Inizio calcolo feature tecniche.")
    df = df_input.copy()

    if 'Close' not in df.columns:
        st.error("[ml_model] ERRORE: Colonna 'Close' non trovata nel DataFrame per il calcolo delle feature.")
        return pd.DataFrame() # Ritorna DataFrame vuoto se manca 'Close'

    # Medie Mobili Semplici (SMA)
    df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['MA50'] = df['Close'].rolling(window=50, min_periods=1).mean()

    # RSI (Relative Strength Index) - Implementazione Semplificata/Placeholder
    # Per un RSI corretto, usare la libreria 'ta' o un'implementazione completa.
    # delta = df['Close'].diff()
    # gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    # loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    # rs = gain / loss
    # df['RSI'] = 100 - (100 / (1 + rs))
    # df['RSI'].fillna(50, inplace=True) # Placeholder per i primi valori
    df['RSI'] = 50.0 # Placeholder fisso per ora

    # Stochastic RSI - Placeholder
    # Richiede RSI. Per un StochRSI corretto, usare la libreria 'ta'.
    # rsi_series = df['RSI'] # Usare l'RSI calcolato sopra
    # min_rsi = rsi_series.rolling(window=14).min()
    # max_rsi = rsi_series.rolling(window=14).max()
    # df['StochRSI_K'] = ((rsi_series - min_rsi) / (max_rsi - min_rsi)) * 100
    # df['StochRSI_D'] = df['StochRSI_K'].rolling(window=3).mean() # Tipico StochRSI %D
    # df['StochRSI_K'].fillna(50, inplace=True) # Placeholder per i primi valori
    # df['StochRSI_D'].fillna(50, inplace=True)
    df['StochRSI'] = 0.5 # Placeholder fisso per ora (valore tra 0 e 1, o 0-100)

    # Momentum - Semplice differenza sul prezzo di chiusura
    df['Momentum'] = df['Close'].diff(periods=1) # Momentum a 1 periodo
    # Oppure, Momentum(N) = Close - Close_N_periodi_fa
    # df['Momentum_10d'] = df['Close'].diff(periods=10)
    
    st.write(f"DEBUG [ml_model]: Feature tecniche calcolate. Shape: {df.shape}")
    # Rimuovi NaN creati dalle rolling windows all'inizio del DataFrame.
    # Il numero di righe da droppare dipende dalla finestra più lunga (es. MA50).
    # Oppure, gestisci i NaN in modo diverso (es. fillna) se preferisci non perdere dati.
    # df.dropna(inplace=True) # Questo potrebbe rimuovere troppi dati se ci sono NaN sparsi.
    # Meglio lasciare che il training gestisca i NaN o che vengano rimossi dopo la creazione dei target.
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
        return df # Ritorna il df originale senza la colonna target

    # Calcola il prezzo futuro e poi la variazione percentuale
    # shift(-horizon) sposta i valori futuri sulle righe attuali
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
    """
    Allena un modello RandomForestRegressor.
    Assicurati che df_features_and_target non contenga NaN nelle feature_columns e target_column.
    """
    st.write(f"DEBUG [ml_model]: Inizio training RandomForest per target '{target_column}'. Feature: {feature_columns}")
    
    df_train = df_features_and_target.dropna(subset=feature_columns + [target_column]) # Rimuove righe con NaN nelle colonne usate

    if df_train.empty:
        st.error("[ml_model] ERRORE: Nessun dato valido per il training dopo la rimozione dei NaN.")
        return None
    
    if not all(col in df_train.columns for col in feature_columns):
        st.error(f"[ml_model] ERRORE: Alcune feature columns ({feature_columns}) non trovate nel DataFrame di training.")
        return None
    if target_column not in df_train.columns:
        st.error(f"[ml_model] ERRORE: Target column '{target_column}' non trovata nel DataFrame di training.")
        return None

    X_train = df_train[feature_columns]
    y_train = df_train[target_column]

    if X_train.empty or len(X_train) != len(y_train):
        st.error("[ml_model] ERRORE: Dati X o y non validi o di lunghezza diversa per il training.")
        return None
        
    model = RandomForestRegressor(
        n_estimators=n_estimators, 
        random_state=random_state, 
        n_jobs=-1, # Usa tutti i processori disponibili
        max_depth=10, # Esempio di iperparametro per evitare overfitting
        min_samples_split=10, # Esempio
        min_samples_leaf=5   # Esempio
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
    """
    Genera predizioni usando il modello allenato.
    Gestisce i NaN nelle feature columns prendendo solo le righe complete per la predizione.
    """
    st.write(f"DEBUG [ml_model]: Inizio generazione predizioni con feature: {feature_columns}")
    
    if df_with_features.empty:
        st.error("[ml_model] ERRORE: DataFrame vuoto fornito per la predizione.")
        return None

    if not all(col in df_with_features.columns for col in feature_columns):
        st.error(f"[ml_model] ERRORE: Alcune feature columns ({feature_columns}) non trovate nel DataFrame per la predizione.")
        return None

    # Seleziona solo le righe dove tutte le feature necessarie sono non-NaN
    df_predict_valid = df_with_features.dropna(subset=feature_columns)
    
    if df_predict_valid.empty:
        st.warning("[ml_model] ATTENZIONE: Nessun dato valido per la predizione dopo la rimozione dei NaN nelle feature.")
        # Restituisce una Series vuota con l'indice originale per mantenere la coerenza se necessario
        return pd.Series(index=df_with_features.index, dtype=float, name="prediction")


    X_predict = df_predict_valid[feature_columns]
    
    try:
        predictions_valid = model.predict(X_predict)
        
        # Crea una Series con le predizioni, allineata con l'indice originale del df_with_features
        # Inizializza con NaN, poi riempi con le predizioni valide
        predictions_aligned = pd.Series(index=df_with_features.index, dtype=float, name="prediction")
        predictions_aligned.loc[df_predict_valid.index] = predictions_valid
        
        st.write(f"DEBUG [ml_model]: Predizioni generate. Numero di predizioni valide: {len(predictions_valid)}. Lunghezza totale Series: {len(predictions_aligned)}")
        return predictions_aligned
    except Exception as e:
        st.error(f"[ml_model] ERRORE durante la generazione delle predizioni: {e}")
        return None


# --- Funzioni per l'integrazione con Google AI Studio (Placeholder) ---
def get_predictions_from_ai_studio(df_features: pd.DataFrame, config: dict) -> pd.Series | None:
    """
    Placeholder per ottenere predizioni da un endpoint di Google AI Studio.
    """
    st.info("[ml_model]: Integrazione Google AI Studio non ancora implementata.")
    # Qui ci sarebbe la logica per chiamare l'API di AI Studio
    # ai_studio_url_secret = config.get('google_ai_studio_url_secret_name')
    # ai_studio_token_secret = config.get('google_ai_studio_token_secret_name')
    # ai_studio_url = st.secrets.get(ai_studio_url_secret)
    # ai_studio_token = st.secrets.get(ai_studio_token_secret)
    # if not ai_studio_url or not ai_studio_token:
    #     st.error(f"[ml_model]: URL ({ai_studio_url_secret}) o Token ({ai_studio_token_secret}) per Google AI Studio mancanti nei secrets.")
    #     return None
    # ... logica della chiamata API con le ultime feature da df_features ...
    # Dovrebbe restituire una Series allineata con l'indice di df_features se possibile
    return pd.Series(index=df_features.index, dtype=float, name="prediction_ai_studio") # Placeholder


if __name__ == '__main__':
    # Blocco per test standalone
    st.write("--- INIZIO TEST STANDALONE ml_model.py ---")

    # Creare un DataFrame di esempio
    sample_data = {
        'Close': [100, 102, 101, 103, 105, 104, 106, 108, 107, 110, 
                  112, 111, 113, 115, 114, 116, 118, 117, 120, 119,
                  122, 121, 123, 125, 124, 126, 128, 127, 130, 129]
    }
    sample_dates = pd.date_range(start='2023-01-01', periods=len(sample_data['Close']), freq='B')
    df_sample = pd.DataFrame(sample_data, index=sample_dates)
    st.write("DataFrame di Esempio Iniziale:")
    st.dataframe(df_sample.head())

    # 1. Test calcolo feature
    df_with_features = calculate_technical_features(df_sample)
    st.write("\nDataFrame con Feature Tecniche:")
    st.dataframe(df_with_features.tail()) # Mostra la coda per vedere i valori calcolati

    # 2. Test creazione target
    prediction_horizon = 3 # giorni
    df_with_target = create_prediction_targets(df_with_features, horizon=prediction_horizon)
    st.write(f"\nDataFrame con Target di Predizione ({prediction_horizon}d % change):")
    st.dataframe(df_with_target.tail(prediction_horizon + 5)) # Mostra la coda, inclusi i NaN del target

    # 3. Test training modello
    # Definisci le colonne delle feature da usare. Devono esistere in df_with_target.
    # Escludi quelle che potrebbero avere troppi NaN o che non sono ancora implementate bene.
    feature_cols_for_training = ['MA20', 'MA50', 'Momentum'] # 'RSI', 'StochRSI' sono placeholder
    target_col_for_training = f'target_{prediction_horizon}d_pct_change'
    
    st.write(f"\nTraining RandomForest con feature: {feature_cols_for_training} e target: {target_col_for_training}")
    trained_rf_model = train_random_forest_model(
        df_with_target,
        feature_columns=feature_cols_for_training,
        target_column=target_col_for_training,
        n_estimators=10 # Pochi stimatori per un test veloce
    )

    if trained_rf_model:
        st.write("Modello RandomForest addestrato con successo (test).")
        
        # 4. Test generazione predizioni
        # Usiamo lo stesso df_with_target per generare predizioni (simulando un backtest)
        predictions_series = generate_model_predictions(
            trained_rf_model,
            df_with_target, # Il DataFrame che ora contiene le feature
            feature_columns=feature_cols_for_training
        )
        if predictions_series is not None:
            st.write("\nSerie di Predizioni Generate:")
            # Unisci le predizioni al DataFrame per visualizzazione
            df_final_test = df_with_target.copy()
            df_final_test['prediction'] = predictions_series
            st.dataframe(df_final_test.tail(10))
        else:
            st.error("Fallita generazione predizioni (test).")
    else:
        st.error("Fallito training del modello RandomForest (test).")
        
    st.write("\n--- FINE TEST STANDALONE ml_model.py ---")
