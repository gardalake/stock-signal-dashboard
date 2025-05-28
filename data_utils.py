# data_utils.py - v1.6.5
import pandas as pd
import requests
import streamlit as st # Per accedere a st.secrets e mostrare messaggi
import time # Per gestire i rate limit di Alpha Vantage (non implementato attivamente ora)

# --- COSTANTI API ---
AV_BASE_URL = "https://www.alphavantage.co/query"
CG_BASE_URL = "https://api.coingecko.com/api/v3"

def get_stock_data(api_key: str, ticker: str, av_function: str, av_outputsize: str) -> pd.DataFrame | None:
    """
    Recupera i dati storici per un ticker azionario da Alpha Vantage.
    """
    if not api_key:
        st.error("[data_utils] ERRORE CRITICO: Chiave API Alpha Vantage non fornita a get_stock_data.")
        return None
        
    params = {
        "function": av_function,
        "symbol": ticker,
        "outputsize": av_outputsize,
        "apikey": api_key,
        "datatype": "json"
    }
    
    st.write(f"DEBUG [data_utils]: Tentativo fetch Alpha Vantage per {ticker} con funzione {av_function}, outputsize {av_outputsize}.")
    try:
        response = requests.get(AV_BASE_URL, params=params, timeout=30) # Timeout aumentato
        response.raise_for_status()
        data = response.json()

        if not data: # Risposta vuota
            st.warning(f"[data_utils] Alpha Vantage: Risposta vuota per {ticker}.")
            return None

        # Gestione messaggi di errore/limite API da Alpha Vantage
        if "Note" in data or "Information" in data or "Error Message" in data:
            msg = data.get("Note", data.get("Information", data.get("Error Message", "Messaggio API Alpha Vantage non riconosciuto.")))
            st.error(f"[data_utils] Alpha Vantage API per {ticker}: {msg}")
            return None
        
        data_key_map = {
            "TIME_SERIES_DAILY_ADJUSTED": "Time Series (Daily)",
            "TIME_SERIES_DAILY": "Time Series (Daily)",
            # Aggiungere mappature per TIME_SERIES_INTRADAY se si useranno dati orari
            # "TIME_SERIES_INTRADAY": f"Time Series ({params.get('interval', '5min')})" 
        }
        actual_data_key = data_key_map.get(av_function)

        if not actual_data_key or actual_data_key not in data:
            st.error(f"[data_utils] Alpha Vantage: Chiave dati '{actual_data_key}' non trovata nella risposta per {ticker}. Funzione: {av_function}. Risposta: {str(data)[:300]}...")
            return None

        df = pd.DataFrame.from_dict(data[actual_data_key], orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index(ascending=True)

        rename_map = {col: col.split(". ")[1].replace(" ", "_").capitalize() if ". " in col else col for col in df.columns}
        df.rename(columns=rename_map, inplace=True)
        
        # Colonne standard OHLCV + Adjusted Close
        ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if 'Adjusted_close' in df.columns: # Alpha Vantage usa 'Adjusted_close'
             df.rename(columns={'Adjusted_close': 'Adj_close'}, inplace=True)
             ohlcv_cols.append('Adj_close')

        for col in ohlcv_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else: # Se manca una colonna OHLCV fondamentale (es. dopo una ridenominazione errata)
                st.warning(f"[data_utils] Alpha Vantage: Colonna attesa '{col}' mancante dopo la ridenominazione per {ticker}.")

        df.dropna(subset=['Close'], inplace=True) # Rimuovi righe se 'Close' è NaN dopo la conversione

        if df.empty:
            st.warning(f"[data_utils] Alpha Vantage: Nessun dato valido per {ticker} dopo la processazione.")
            return None

        st.write(f"DEBUG [data_utils]: Dati per {ticker} caricati da Alpha Vantage. Shape: {df.shape}")
        return df

    except requests.exceptions.Timeout:
        st.error(f"[data_utils] Alpha Vantage: Timeout durante il caricamento dati per {ticker}.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"[data_utils] Alpha Vantage: Errore HTTP per {ticker}: {e}. Response: {e.response.text[:200] if e.response else 'N/A'}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"[data_utils] Alpha Vantage: Errore di richiesta generico per {ticker}: {e}")
        return None
    except ValueError as e: # Include JSONDecodeError
        st.error(f"[data_utils] Alpha Vantage: Errore nel decodificare i dati JSON per {ticker}: {e}. Risposta (primi 200 caratteri): {response.text[:200] if 'response' in locals() else 'N/A'}")
        return None
    except Exception as e:
        st.error(f"[data_utils] Alpha Vantage: Errore imprevisto per {ticker}: {e}")
        # import traceback
        # st.text(traceback.format_exc()) # Per debug più approfondito
        return None


def get_crypto_data(coin_id: str, vs_currency: str, days: int) -> pd.DataFrame | None:
    """
    Recupera i dati storici OHLCV per una criptovaluta da CoinGecko.
    Usa l'endpoint /coins/{id}/ohlc.
    """
    params = {
        "vs_currency": vs_currency,
        "days": str(days),
    }
    url = f"{CG_BASE_URL}/coins/{coin_id}/ohlc"
    
    st.write(f"DEBUG [data_utils]: Tentativo fetch CoinGecko OHLC per {coin_id}/{vs_currency} per {days} giorni.")
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data_ohlc = response.json() # Restituisce una lista di liste: [timestamp, open, high, low, close]

        if not data_ohlc:
            st.warning(f"[data_utils] CoinGecko: Nessun dato OHLC restituito per {coin_id}.")
            return None

        df = pd.DataFrame(data_ohlc, columns=['Timestamp', 'Open', 'High', 'Low', 'Close'])
        df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
        df.set_index('Date', inplace=True)
        df.drop(columns=['Timestamp'], inplace=True)
        
        # CoinGecko per OHLC non fornisce il volume direttamente in questo endpoint.
        # Per il volume, si dovrebbe fare una seconda chiamata a /coins/{id}/market_chart
        # o accettare che il volume non sia disponibile con questo singolo endpoint.
        # Per ora, aggiungiamo una colonna Volume fittizia o la omettiamo.
        # Aggiungiamo una colonna Volume con 0 per coerenza con le specifiche.
        df['Volume'] = 0.0 
        
        # Per le crypto, Adj_close è solitamente uguale a Close
        df['Adj_close'] = df['Close']

        # Assicurati che le colonne siano numeriche
        for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj_close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=['Close'], inplace=True)

        if df.empty:
            st.warning(f"[data_utils] CoinGecko: Nessun dato valido per {coin_id} dopo la processazione.")
            return None

        st.write(f"DEBUG [data_utils]: Dati per {coin_id} caricati da CoinGecko (OHLC). Shape: {df.shape}")
        return df

    except requests.exceptions.Timeout:
        st.error(f"[data_utils] CoinGecko: Timeout durante il caricamento dati per {coin_id}.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"[data_utils] CoinGecko: Errore HTTP per {coin_id}: {e}. Response: {e.response.text[:200] if e.response else 'N/A'}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"[data_utils] CoinGecko: Errore di richiesta generico per {coin_id}: {e}")
        return None
    except ValueError as e: # Include JSONDecodeError
        st.error(f"[data_utils] CoinGecko: Errore nel decodificare i dati JSON per {coin_id}: {e}. Risposta (primi 200 caratteri): {response.text[:200] if 'response' in locals() else 'N/A'}")
        return None
    except Exception as e:
        st.error(f"[data_utils] CoinGecko: Errore imprevisto per {coin_id}: {e}")
        # import traceback
        # st.text(traceback.format_exc())
        return None

# Blocco per test standalone (eseguire `python data_utils.py`)
# Richiede di gestire le secrets diversamente se non eseguito tramite Streamlit.
if __name__ == '__main__':
    st.write("--- INIZIO TEST STANDALONE data_utils.py ---")
    
    # Per testare Alpha Vantage, avresti bisogno di una chiave API.
    # Poiché st.secrets non è disponibile qui, dovresti caricarla manualmente.
    # Esempio: TEST_AV_API_KEY = "LA_TUA_CHIAVE_QUI" (NON COMMETTERE!)
    TEST_AV_API_KEY = None # Disabilitato di default per sicurezza
    
    if TEST_AV_API_KEY:
        st.write("\n--- Test Alpha Vantage ---")
        df_stock = get_stock_data(TEST_AV_API_KEY, "IBM", "TIME_SERIES_DAILY_ADJUSTED", "compact")
        if df_stock is not None:
            st.write("Dati IBM (Alpha Vantage):")
            st.dataframe(df_stock.tail())
        else:
            st.write("Fallito caricamento dati IBM (Alpha Vantage).")
    else:
        st.write("\nTest Alpha Vantage saltato: TEST_AV_API_KEY non impostata in data_utils.py per test standalone.")

    st.write("\n--- Test CoinGecko ---")
    # Test CoinGecko per Bitcoin
    df_crypto_btc = get_crypto_data(coin_id="bitcoin", vs_currency="usd", days=30)
    if df_crypto_btc is not None:
        st.write("Dati Bitcoin (CoinGecko OHLC):")
        st.dataframe(df_crypto_btc.tail())
    else:
        st.write("Fallito caricamento dati Bitcoin (CoinGecko).")

    # Test CoinGecko per un'altra crypto, es. Ethereum
    df_crypto_eth = get_crypto_data(coin_id="ethereum", vs_currency="usd", days=7)
    if df_crypto_eth is not None:
        st.write("\nDati Ethereum (CoinGecko OHLC):")
        st.dataframe(df_crypto_eth.tail())
    else:
        st.write("Fallito caricamento dati Ethereum (CoinGecko).")
    
    st.write("\n--- FINE TEST STANDALONE data_utils.py ---")
