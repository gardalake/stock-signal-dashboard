# data_utils.py - v1.6.5 (Crypto data loading improved + Caching)
import pandas as pd
import requests
import streamlit as st 
from datetime import datetime, timedelta 

# --- COSTANTI API ---
AV_BASE_URL = "https://www.alphavantage.co/query"
CG_BASE_URL = "https://api.coingecko.com/api/v3"

DATA_CACHE_TTL = 900 # 15 minuti (900 secondi)

@st.cache_data(ttl=DATA_CACHE_TTL, show_spinner=f"Caricamento dati Alpha Vantage...") # Mostra uno spinner per la cache
def get_stock_data(api_key: str, ticker: str, av_function: str, av_outputsize: str) -> pd.DataFrame | None:
    """
    Recupera i dati storici per un ticker azionario da Alpha Vantage.
    """
    # Usare print per i log interni alle funzioni cachate, appariranno nei log del server.
    print(f"INFO [data_utils.get_stock_data]: Chiamata API (o cache) per {ticker}, func: {av_function}, size: {av_outputsize}")
    if not api_key:
        print(f"ERROR [data_utils.get_stock_data]: Chiave API Alpha Vantage non fornita.")
        return None
        
    params = {
        "function": av_function,
        "symbol": ticker,
        "outputsize": av_outputsize,
        "apikey": api_key,
        "datatype": "json"
    }
    
    print(f"DEBUG [data_utils.get_stock_data]: Tentativo fetch Alpha Vantage per {ticker}.")
    try:
        response = requests.get(AV_BASE_URL, params=params, timeout=30) 
        response.raise_for_status()
        data = response.json()

        if not data: 
            print(f"WARNING [data_utils.get_stock_data]: Alpha Vantage: Risposta vuota per {ticker}.")
            return None

        if "Note" in data or "Information" in data or "Error Message" in data:
            msg = data.get("Note", data.get("Information", data.get("Error Message", "Messaggio API Alpha Vantage non riconosciuto.")))
            print(f"ERROR [data_utils.get_stock_data]: Alpha Vantage API per {ticker}: {msg}")
            return None 
        
        data_key_map = {
            "TIME_SERIES_DAILY_ADJUSTED": "Time Series (Daily)",
            "TIME_SERIES_DAILY": "Time Series (Daily)",
        }
        actual_data_key = data_key_map.get(av_function)

        if not actual_data_key or actual_data_key not in data:
            print(f"ERROR [data_utils.get_stock_data]: Alpha Vantage: Chiave dati '{actual_data_key}' non trovata per {ticker}. Funzione: {av_function}. Risposta: {str(data)[:300]}...")
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
                print(f"WARNING [data_utils.get_stock_data]: Alpha Vantage: Colonna '{col}' mancante per {ticker}.")

        df.dropna(subset=['Close'], inplace=True) 

        if df.empty:
            print(f"WARNING [data_utils.get_stock_data]: Alpha Vantage: Nessun dato valido per {ticker} dopo processazione.")
            return None

        print(f"DEBUG [data_utils.get_stock_data]: Dati per {ticker} caricati da Alpha Vantage. Shape: {df.shape}")
        return df

    except requests.exceptions.Timeout:
        print(f"ERROR [data_utils.get_stock_data]: Alpha Vantage: Timeout per {ticker}.")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"ERROR [data_utils.get_stock_data]: Alpha Vantage: Errore HTTP per {ticker}: {e}. Response: {e.response.text[:200] if e.response else 'N/A'}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"ERROR [data_utils.get_stock_data]: Alpha Vantage: Errore richiesta per {ticker}: {e}")
        return None
    except ValueError as e: 
        response_text_for_error = response.text[:200] if 'response' in locals() and hasattr(response, 'text') else 'N/A'
        print(f"ERROR [data_utils.get_stock_data]: Alpha Vantage: Errore JSON per {ticker}: {e}. Risposta: {response_text_for_error}")
        return None
    except Exception as e:
        print(f"ERROR [data_utils.get_stock_data]: Alpha Vantage: Errore imprevisto per {ticker}: {e}")
        return None

def _aggregate_hourly_to_daily_ohlc(df_hourly: pd.DataFrame) -> pd.DataFrame:
    if df_hourly.empty or not isinstance(df_hourly.index, pd.DatetimeIndex):
        return pd.DataFrame()
    df_hourly.index = pd.to_datetime(df_hourly.index)
    aggregation_rules = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    cols_to_keep = [col for col in aggregation_rules.keys() if col in df_hourly.columns]
    if not cols_to_keep:
        print("WARNING [data_utils._aggregate_hourly_to_daily_ohlc]: Nessuna colonna valida per aggregazione.")
        return pd.DataFrame()
    df_daily = df_hourly[cols_to_keep].resample('D').agg(aggregation_rules)
    df_daily.dropna(how='all', inplace=True) 
    if 'Close' in df_daily.columns:
        df_daily['Adj_close'] = df_daily['Close']
    return df_daily

@st.cache_data(ttl=DATA_CACHE_TTL, show_spinner=f"Caricamento dati CoinGecko...") # Mostra uno spinner per la cache
def get_crypto_data(coin_id: str, vs_currency: str, days: int) -> pd.DataFrame | None:
    """
    Recupera i dati storici per una criptovaluta da CoinGecko.
    Tenta di usare /ohlc per <= 90 giorni.
    Per > 90 giorni (o se /ohlc fallisce), usa /market_chart.
    """
    print(f"INFO [data_utils.get_crypto_data]: Chiamata API (o cache) per {coin_id}, vs_curr: {vs_currency}, days: {days}")
    df_final = None
    use_ohlc_endpoint = days <= 90 

    if use_ohlc_endpoint:
        print(f"DEBUG [data_utils.get_crypto_data]: Tentativo fetch CoinGecko /ohlc (giorni={days}) per {coin_id}.")
        params_ohlc = {"vs_currency": vs_currency, "days": str(days)}
        url_ohlc = f"{CG_BASE_URL}/coins/{coin_id}/ohlc"
        try:
            response_ohlc = requests.get(url_ohlc, params=params_ohlc, timeout=20)
            response_ohlc.raise_for_status()
            data_ohlc = response_ohlc.json()

            if data_ohlc:
                df_temp = pd.DataFrame(data_ohlc, columns=['Timestamp', 'Open', 'High', 'Low', 'Close'])
                df_temp['Date'] = pd.to_datetime(df_temp['Timestamp'], unit='ms')
                df_temp.set_index('Date', inplace=True)
                df_temp.drop(columns=['Timestamp'], inplace=True)
                
                params_market_chart_vol = {"vs_currency": vs_currency, "days": str(days), "interval": "daily"}
                url_market_chart_vol = f"{CG_BASE_URL}/coins/{coin_id}/market_chart"
                print(f"DEBUG [data_utils.get_crypto_data]: Tentativo fetch CoinGecko market_chart per VOLUME per {coin_id}.")
                response_vol = requests.get(url_market_chart_vol, params=params_market_chart_vol, timeout=20)
                response_vol.raise_for_status()
                data_market_chart_vol = response_vol.json()

                if 'total_volumes' in data_market_chart_vol and data_market_chart_vol['total_volumes']:
                    df_volumes = pd.DataFrame(data_market_chart_vol['total_volumes'], columns=['Timestamp', 'Volume'])
                    df_volumes['Date'] = pd.to_datetime(df_volumes['Timestamp'], unit='ms').dt.normalize() 
                    df_volumes.set_index('Date', inplace=True)
                    df_temp = df_temp.merge(df_volumes[['Volume']], left_index=True, right_index=True, how='left')
                    df_temp['Volume'].fillna(0, inplace=True) 
                else:
                    df_temp['Volume'] = 0.0
                
                df_final = df_temp
                print(f"DEBUG [data_utils.get_crypto_data]: Dati da CoinGecko /ohlc (+ volume). Shape: {df_final.shape if df_final is not None else 'None'}")
            else:
                print(f"WARNING [data_utils.get_crypto_data]: CoinGecko /ohlc: Nessun dato per {coin_id} per {days} giorni.")
        except Exception as e:
            print(f"WARNING [data_utils.get_crypto_data]: CoinGecko /ohlc: Errore (tentativo con /market_chart sotto): {e}")
    
    if df_final is None or days > 90 : 
        if days > 90 :
             print(f"DEBUG [data_utils.get_crypto_data]: Richiesti {days} giorni (>90), usando /market_chart per {coin_id}.")
        elif df_final is None: 
             print(f"DEBUG [data_utils.get_crypto_data]: /ohlc fallito, usando /market_chart per {coin_id} ({days} giorni).")

        params_market_chart = {"vs_currency": vs_currency, "days": str(days), "interval": "daily"}
        url_market_chart = f"{CG_BASE_URL}/coins/{coin_id}/market_chart"
        try:
            response_mc = requests.get(url_market_chart, params=params_market_chart, timeout=30)
            response_mc.raise_for_status()
            data_market = response_mc.json()

            if 'prices' in data_market and data_market['prices']:
                df_prices = pd.DataFrame(data_market['prices'], columns=['Timestamp', 'Close'])
                df_prices['Date'] = pd.to_datetime(df_prices['Timestamp'], unit='ms').dt.normalize()
                df_prices.set_index('Date', inplace=True)
                df_prices.drop(columns=['Timestamp'], inplace=True)

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
                
                df_final = df_final_mc[['Open', 'High', 'Low', 'Close', 'Volume']] 
                print(f"DEBUG [data_utils.get_crypto_data]: Dati da CoinGecko /market_chart. Shape: {df_final.shape if df_final is not None else 'None'}")
            else:
                print(f"WARNING [data_utils.get_crypto_data]: CoinGecko /market_chart: Nessun dato 'prices' per {coin_id}.")
                return None 

        except requests.exceptions.HTTPError as e_mc: # Catch HTTPError specificamente per il rate limit
            if e_mc.response.status_code == 429:
                print(f"ERROR [data_utils.get_crypto_data]: CoinGecko /market_chart: Rate limit (429 Too Many Requests) per {coin_id}. Dettagli: {e_mc.response.text[:200] if e_mc.response else 'N/A'}")
            else:
                print(f"ERROR [data_utils.get_crypto_data]: CoinGecko /market_chart: Errore HTTP per {coin_id}: {e_mc}. Response: {e_mc.response.text[:200] if e_mc.response else 'N/A'}")
            return None
        except Exception as e_mc:
            response_text_for_error_mc = response_mc.text[:200] if 'response_mc' in locals() and hasattr(response_mc, 'text') else 'N/A'
            print(f"ERROR [data_utils.get_crypto_data]: CoinGecko /market_chart: Errore caricamento per {coin_id}: {e_mc}. Risposta: {response_text_for_error_mc}")
            return None

    if df_final is None or df_final.empty:
        print(f"WARNING [data_utils.get_crypto_data]: CoinGecko: Nessun dato valido per {coin_id} dopo tutti i tentativi.")
        return None

    if 'Close' in df_final.columns:
        df_final['Adj_close'] = df_final['Close'] 
    
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj_close']
    for col in numeric_cols:
        if col in df_final.columns:
            df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
    
    df_final.dropna(subset=['Close'], inplace=True) 

    if df_final.empty:
        print(f"WARNING [data_utils.get_crypto_data]: CoinGecko: Nessun dato valido per {coin_id} dopo processazione finale.")
        return None
    
    df_final = df_final.sort_index(ascending=True)
    print(f"DEBUG [data_utils.get_crypto_data]: Dati finali per {coin_id} caricati. Shape: {df_final.shape}")
    return df_final


if __name__ == '__main__':
    print("--- INIZIO TEST STANDALONE data_utils.py ---") # Cambiato st.write a print
    
    TEST_AV_API_KEY = None 
    if TEST_AV_API_KEY:
        print("\n--- Test Alpha Vantage ---")
        df_stock = get_stock_data(TEST_AV_API_KEY, "IBM", "TIME_SERIES_DAILY_ADJUSTED", "compact")
        if df_stock is not None:
            print("Dati IBM (Alpha Vantage):")
            print(df_stock.tail().to_string()) # Usare print e to_string per DataFrame
    else:
        print("\nTest Alpha Vantage saltato: TEST_AV_API_KEY non impostata.")

    print("\n--- Test CoinGecko ---")
    print("Test per Bitcoin (<=90 giorni):")
    df_crypto_btc_short = get_crypto_data(coin_id="bitcoin", vs_currency="usd", days=30)
    if df_crypto_btc_short is not None:
        print(df_crypto_btc_short.tail().to_string())
    
    print("\nTest per Ethereum (>90 giorni):")
    df_crypto_eth_long = get_crypto_data(coin_id="ethereum", vs_currency="usd", days=120) 
    if df_crypto_eth_long is not None:
        print(df_crypto_eth_long.tail().to_string())
    
    print("\n--- FINE TEST STANDALONE data_utils.py ---")
