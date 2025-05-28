# data_utils.py - v1.6.5 (Crypto data loading improved)
import pandas as pd
import requests
import streamlit as st # Per accedere a st.secrets e mostrare messaggi
import time # Per gestire i rate limit di Alpha Vantage (non implementato attivamente ora)
from datetime import datetime, timedelta # Per la gestione delle date in get_crypto_data

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
        response = requests.get(AV_BASE_URL, params=params, timeout=30) 
        response.raise_for_status()
        data = response.json()

        if not data: 
            st.warning(f"[data_utils] Alpha Vantage: Risposta vuota per {ticker}.")
            return None

        if "Note" in data or "Information" in data or "Error Message" in data:
            msg = data.get("Note", data.get("Information", data.get("Error Message", "Messaggio API Alpha Vantage non riconosciuto.")))
            st.error(f"[data_utils] Alpha Vantage API per {ticker}: {msg}")
            return None
        
        data_key_map = {
            "TIME_SERIES_DAILY_ADJUSTED": "Time Series (Daily)",
            "TIME_SERIES_DAILY": "Time Series (Daily)",
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
        
        ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if 'Adjusted_close' in df.columns: 
             df.rename(columns={'Adjusted_close': 'Adj_close'}, inplace=True)
             ohlcv_cols.append('Adj_close')

        for col in ohlcv_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else: 
                st.warning(f"[data_utils] Alpha Vantage: Colonna attesa '{col}' mancante dopo la ridenominazione per {ticker}.")

        df.dropna(subset=['Close'], inplace=True) 

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
    except ValueError as e: 
        st.error(f"[data_utils] Alpha Vantage: Errore nel decodificare i dati JSON per {ticker}: {e}. Risposta (primi 200 caratteri): {response.text[:200] if 'response' in locals() else 'N/A'}")
        return None
    except Exception as e:
        st.error(f"[data_utils] Alpha Vantage: Errore imprevisto per {ticker}: {e}")
        return None


def _aggregate_hourly_to_daily_ohlc(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """Helper per aggregare dati OHLC orari a giornalieri."""
    if df_hourly.empty or not isinstance(df_hourly.index, pd.DatetimeIndex):
        return pd.DataFrame()

    # Assicura che l'indice sia datetime e normalizzato a giorno per il groupby
    df_hourly.index = pd.to_datetime(df_hourly.index)
    
    aggregation_rules = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum' # Somma il volume orario per ottenere il volume giornaliero
    }
    
    # Rimuovi colonne non necessarie prima dell'aggregazione se presenti
    cols_to_keep = [col for col in aggregation_rules.keys() if col in df_hourly.columns]
    if not cols_to_keep:
        st.warning("[data_utils] _aggregate_hourly_to_daily_ohlc: Nessuna colonna valida (Open, High, Low, Close, Volume) trovata per l'aggregazione.")
        return pd.DataFrame()

    df_daily = df_hourly[cols_to_keep].resample('D').agg(aggregation_rules)
    df_daily.dropna(how='all', inplace=True) # Rimuovi giorni senza dati (weekend, festivi)
    
    if 'Close' in df_daily.columns: # Aggiungi Adj_close se Close esiste
        df_daily['Adj_close'] = df_daily['Close']
    
    return df_daily


def get_crypto_data(coin_id: str, vs_currency: str, days: int) -> pd.DataFrame | None:
    """
    Recupera i dati storici per una criptovaluta da CoinGecko.
    Tenta di usare /ohlc per <= 90 giorni per dati OHLC giornalieri.
    Per > 90 giorni, usa /market_chart e inferisce OHLC (O,H,L = C) da dati giornalieri.
    """
    st.write(f"DEBUG [data_utils]: Richiesta dati CoinGecko per {coin_id}/{vs_currency} per {days} giorni.")
    df_final = None

    if days <= 90:
        st.write(f"DEBUG [data_utils]: Tentativo fetch CoinGecko OHLC (<=90 giorni) per {coin_id}.")
        params_ohlc = {"vs_currency": vs_currency, "days": str(days)}
        url_ohlc = f"{CG_BASE_URL}/coins/{coin_id}/ohlc"
        try:
            response = requests.get(url_ohlc, params=params_ohlc, timeout=20)
            response.raise_for_status()
            data_ohlc = response.json()

            if data_ohlc:
                df_temp = pd.DataFrame(data_ohlc, columns=['Timestamp', 'Open', 'High', 'Low', 'Close'])
                df_temp['Date'] = pd.to_datetime(df_temp['Timestamp'], unit='ms')
                df_temp.set_index('Date', inplace=True)
                df_temp.drop(columns=['Timestamp'], inplace=True)
                
                # Endpoint OHLC non fornisce volume; richiediamo market_chart per il volume
                # Questo aggiunge una chiamata API, ma è necessario per il volume
                params_market_chart_vol = {"vs_currency": vs_currency, "days": str(days), "interval": "daily"}
                url_market_chart_vol = f"{CG_BASE_URL}/coins/{coin_id}/market_chart"
                st.write(f"DEBUG [data_utils]: Tentativo fetch CoinGecko market_chart per VOLUME per {coin_id}.")
                response_vol = requests.get(url_market_chart_vol, params=params_market_chart_vol, timeout=20)
                response_vol.raise_for_status()
                data_market_chart_vol = response_vol.json()

                if 'total_volumes' in data_market_chart_vol and data_market_chart_vol['total_volumes']:
                    df_volumes = pd.DataFrame(data_market_chart_vol['total_volumes'], columns=['Timestamp', 'Volume'])
                    df_volumes['Date'] = pd.to_datetime(df_volumes['Timestamp'], unit='ms').dt.normalize() # Normalizza a mezzanotte
                    df_volumes.set_index('Date', inplace=True)
                    # Unisci il volume al DataFrame OHLC
                    df_temp = df_temp.merge(df_volumes[['Volume']], left_index=True, right_index=True, how='left')
                    df_temp['Volume'].fillna(0, inplace=True) # Riempi eventuali NaN nel volume
                else:
                    df_temp['Volume'] = 0.0
                
                df_final = df_temp
                st.write(f"DEBUG [data_utils]: Dati da CoinGecko /ohlc (+ volume da /market_chart). Shape: {df_final.shape}")
            else:
                st.warning(f"[data_utils] CoinGecko /ohlc: Nessun dato restituito per {coin_id} per {days} giorni.")
        except Exception as e:
            st.warning(f"[data_utils] CoinGecko /ohlc: Errore durante il caricamento (tentativo con /market_chart): {e}")
            # Se /ohlc fallisce o dà errore, proviamo comunque /market_chart sotto
    
    # Se giorni > 90 o se /ohlc ha fallito e df_final è ancora None
    if df_final is None or days > 90 : # Aggiunto 'days > 90' per forzare market_chart su periodi lunghi
        if days > 90 :
             st.write(f"DEBUG [data_utils]: Richiesti {days} giorni (>90), si passa a CoinGecko /market_chart per {coin_id}.")
        elif df_final is None: # days <=90 ma ohlc ha fallito
             st.write(f"DEBUG [data_utils]: /ohlc fallito o nessun dato, si passa a CoinGecko /market_chart per {coin_id} ({days} giorni).")

        params_market_chart = {"vs_currency": vs_currency, "days": str(days), "interval": "daily"}
        url_market_chart = f"{CG_BASE_URL}/coins/{coin_id}/market_chart"
        try:
            response = requests.get(url_market_chart, params=params_market_chart, timeout=30) # Timeout più lungo per più dati
            response.raise_for_status()
            data_market = response.json()

            if 'prices' in data_market and data_market['prices']:
                df_prices = pd.DataFrame(data_market['prices'], columns=['Timestamp', 'Close'])
                df_prices['Date'] = pd.to_datetime(df_prices['Timestamp'], unit='ms').dt.normalize()
                df_prices.set_index('Date', inplace=True)
                df_prices.drop(columns=['Timestamp'], inplace=True)

                # Inferisci OHLC da Close
                df_final_mc = df_prices.copy()
                df_final_mc['Open'] = df_final_mc['Close']
                df_final_mc['High'] = df_final_mc['Close']
                df_final_mc['Low'] = df_final_mc['Close']

                if 'total_volumes' in data_market and data_market['total_volumes']:
                    df_volumes = pd.DataFrame(data_market['total_volumes'], columns=['Timestamp', 'Volume'])
                    df_volumes['Date'] = pd.to_datetime(df_volumes['Timestamp'], unit='ms').dt.normalize()
                    df_volumes.set_index('Date', inplace=True)
                    df_final_mc = df_final_mc.merge(df_volumes[['Volume']], left_index=True, right_index=True, how='left')
                    df_final_mc['Volume'].fillna(0, inplace=True)
                else:
                    df_final_mc['Volume'] = 0.0
                
                df_final = df_final_mc[['Open', 'High', 'Low', 'Close', 'Volume']] # Riordina colonne
                st.write(f"DEBUG [data_utils]: Dati da CoinGecko /market_chart. Shape: {df_final.shape}")
            else:
                st.warning(f"[data_utils] CoinGecko /market_chart: Nessun dato 'prices' restituito per {coin_id}.")
                return None # Ritorna None se market_chart non dà prezzi

        except Exception as e:
            st.error(f"[data_utils] CoinGecko /market_chart: Errore durante il caricamento per {coin_id}: {e}")
            return None

    if df_final is None or df_final.empty:
        st.warning(f"[data_utils] CoinGecko: Nessun dato valido per {coin_id} dopo tutti i tentativi.")
        return None

    # Standardizzazione finale
    if 'Close' in df_final.columns:
        df_final['Adj_close'] = df_final['Close'] # Per crypto, Adj_close è spesso uguale a Close
    
    # Assicura che le colonne siano numeriche
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj_close']
    for col in numeric_cols:
        if col in df_final.columns:
            df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
    
    df_final.dropna(subset=['Close'], inplace=True) # Rimuovi righe se 'Close' è NaN

    if df_final.empty:
        st.warning(f"[data_utils] CoinGecko: Nessun dato valido per {coin_id} dopo la processazione finale.")
        return None
    
    df_final = df_final.sort_index(ascending=True)
    st.write(f"DEBUG [data_utils]: Dati finali per {coin_id} caricati da CoinGecko. Shape: {df_final.shape}")
    return df_final


if __name__ == '__main__':
    st.write("--- INIZIO TEST STANDALONE data_utils.py ---")
    
    TEST_AV_API_KEY = None 
    if TEST_AV_API_KEY:
        st.write("\n--- Test Alpha Vantage ---")
        df_stock = get_stock_data(TEST_AV_API_KEY, "IBM", "TIME_SERIES_DAILY_ADJUSTED", "compact")
        if df_stock is not None:
            st.write("Dati IBM (Alpha Vantage):")
            st.dataframe(df_stock.tail())
    else:
        st.write("\nTest Alpha Vantage saltato: TEST_AV_API_KEY non impostata.")

    st.write("\n--- Test CoinGecko ---")
    st.write("Test per Bitcoin (<=90 giorni):")
    df_crypto_btc_short = get_crypto_data(coin_id="bitcoin", vs_currency="usd", days=30)
    if df_crypto_btc_short is not None:
        st.dataframe(df_crypto_btc_short.tail())
    
    st.write("\nTest per Ethereum (>90 giorni):")
    df_crypto_eth_long = get_crypto_data(coin_id="ethereum", vs_currency="usd", days=120) # Chiedi più di 90 giorni
    if df_crypto_eth_long is not None:
        st.dataframe(df_crypto_eth_long.tail())
    
    st.write("\n--- FINE TEST STANDALONE data_utils.py ---")
