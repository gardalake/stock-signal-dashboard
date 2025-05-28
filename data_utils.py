# data_utils.py - v1.6.6 (Intraday data handling, Caching, Python Logger)
import pandas as pd
import requests
import streamlit as st 
from datetime import datetime, timedelta 

from logger_utils import setup_logger
logger = setup_logger(__name__) 

AV_BASE_URL = "https://www.alphavantage.co/query"
CG_BASE_URL = "https://api.coingecko.com/api/v3"
DATA_CACHE_TTL = 900 

@st.cache_data(ttl=DATA_CACHE_TTL, show_spinner="Caricamento dati Alpha Vantage...") 
def get_stock_data(
    api_key: str, 
    ticker: str, 
    av_function: str, # Es. TIME_SERIES_DAILY_ADJUSTED o TIME_SERIES_INTRADAY
    av_outputsize: str, 
    av_interval: str | None = None # Es. "1min", "5min", "15min", "30min", "60min" - solo per INTRADAY
) -> pd.DataFrame | None:
    logger.info(f"Chiamata API AV (cache) per {ticker}, func: {av_function}, size: {av_outputsize}, interval: {av_interval if av_interval else 'N/A'}")
    if not api_key:
        logger.error("Chiave API Alpha Vantage non fornita.")
        return None
        
    params = {
        "function": av_function,
        "symbol": ticker,
        "apikey": api_key,
        "datatype": "json" # CSV è un'opzione per alcuni endpoint, ma JSON è più strutturato
    }
    # Aggiungi outputsize e interval solo se sono rilevanti per la funzione e forniti
    if av_function == "TIME_SERIES_INTRADAY":
        if not av_interval:
            logger.error(f"Funzione INTRADAY richiesta per {ticker} ma nessun intervallo specificato.")
            return None
        params["interval"] = av_interval
        params["outputsize"] = av_outputsize # Per intraday, 'outputsize' può essere 'compact' o 'full'
        # Nota: 'extended_hours=false' (default) per evitare dati pre/post market se non desiderati.
        # params["extended_hours"] = "false" 
    elif av_function in ["TIME_SERIES_DAILY_ADJUSTED", "TIME_SERIES_DAILY"]:
        params["outputsize"] = av_outputsize
    # Altre funzioni AV potrebbero non usare outputsize o interval
    
    logger.debug(f"Tentativo fetch Alpha Vantage per {ticker} con parametri: {params}")
    try:
        response = requests.get(AV_BASE_URL, params=params, timeout=45) # Timeout leggermente aumentato per intraday full
        response.raise_for_status()
        data = response.json()

        if not data: 
            logger.warning(f"Alpha Vantage: Risposta vuota per {ticker} con params {params}.")
            return None

        if "Note" in data or "Information" in data or "Error Message" in data:
            msg = data.get("Note", data.get("Information", data.get("Error Message", "Messaggio API Alpha Vantage non riconosciuto.")))
            logger.error(f"Alpha Vantage API per {ticker} ({av_function}): {msg}")
            return None 
        
        # Determina la chiave corretta per i dati nel JSON di risposta
        actual_data_key = None
        if av_function == "TIME_SERIES_INTRADAY" and av_interval:
            actual_data_key = f"Time Series ({av_interval})"
        elif av_function == "TIME_SERIES_DAILY_ADJUSTED":
            actual_data_key = "Time Series (Daily)"
        elif av_function == "TIME_SERIES_DAILY":
            actual_data_key = "Time Series (Daily)"
        # Aggiungere altre mappature se si usano altre funzioni AV

        if not actual_data_key or actual_data_key not in data:
            logger.error(f"Alpha Vantage: Chiave dati '{actual_data_key}' non trovata per {ticker}. Params: {params}. Risposta: {str(data)[:300]}...")
            return None

        df = pd.DataFrame.from_dict(data[actual_data_key], orient="index")
        df.index = pd.to_datetime(df.index) # L'indice sarà DatetimeIndex con data e ora per intraday
        df = df.sort_index(ascending=True)

        rename_map = {col: col.split(". ")[1].replace(" ", "_").capitalize() if ". " in col else col for col in df.columns}
        df.rename(columns=rename_map, inplace=True)
        
        ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if 'Adjusted_close' in df.columns: # Per TIME_SERIES_DAILY_ADJUSTED
             df.rename(columns={'Adjusted_close': 'Adj_close'}, inplace=True)
             ohlcv_cols.append('Adj_close')
        elif av_function != "TIME_SERIES_INTRADAY" and 'Close' in df.columns : # Se non intraday e manca Adj_close, usa Close
            df['Adj_close'] = df['Close']
            ohlcv_cols.append('Adj_close')


        for col in ohlcv_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else: 
                # Per intraday, adjusted close e dividend/split non sono tipicamente forniti
                if not (av_function == "TIME_SERIES_INTRADAY" and col in ['Adj_close', 'Dividend_amount', 'Split_coefficient']):
                    logger.warning(f"Alpha Vantage: Colonna '{col}' mancante per {ticker} ({av_function}).")

        df.dropna(subset=['Close'], inplace=True) 

        if df.empty:
            logger.warning(f"Alpha Vantage: Nessun dato valido per {ticker} ({av_function}) dopo processazione.")
            return None

        logger.debug(f"Dati per {ticker} ({av_function}) caricati da AV. Shape: {df.shape}")
        return df

    except requests.exceptions.Timeout:
        logger.error(f"Alpha Vantage: Timeout per {ticker} ({av_function}).")
        return None
    except requests.exceptions.HTTPError as e:
        logger.error(f"Alpha Vantage: Errore HTTP per {ticker} ({av_function}): {e}. Response: {e.response.text[:200] if e.response else 'N/A'}")
        return None
    # ... (altri blocchi except come prima, aggiungendo av_function ai log) ...
    except requests.exceptions.RequestException as e:
        logger.error(f"Alpha Vantage: Errore richiesta per {ticker} ({av_function}): {e}", exc_info=True)
        return None
    except ValueError as e: 
        response_text_for_error = response.text[:200] if 'response' in locals() and hasattr(response, 'text') else 'N/A'
        logger.error(f"Alpha Vantage: Errore JSON per {ticker} ({av_function}): {e}. Risposta: {response_text_for_error}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Alpha Vantage: Errore imprevisto per {ticker} ({av_function}): {e}", exc_info=True)
        return None

# _aggregate_hourly_to_daily_ohlc (invariato, ma lo includo per completezza)
def _aggregate_hourly_to_daily_ohlc(df_hourly: pd.DataFrame) -> pd.DataFrame:
    if df_hourly.empty or not isinstance(df_hourly.index, pd.DatetimeIndex):
        return pd.DataFrame()
    df_hourly.index = pd.to_datetime(df_hourly.index)
    aggregation_rules = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    cols_to_keep = [col for col in aggregation_rules.keys() if col in df_hourly.columns]
    if not cols_to_keep:
        logger.warning("_aggregate_hourly_to_daily_ohlc: Nessuna colonna valida per aggregazione.")
        return pd.DataFrame()
    df_daily = df_hourly[cols_to_keep].resample('D').agg(aggregation_rules)
    df_daily.dropna(how='all', inplace=True) 
    if 'Close' in df_daily.columns:
        df_daily['Adj_close'] = df_daily['Close']
    return df_daily

@st.cache_data(ttl=DATA_CACHE_TTL, show_spinner="Caricamento dati CoinGecko...") 
def get_crypto_data(
    coin_id: str, 
    vs_currency: str, 
    days: int, 
    # Aggiungiamo un parametro per l'intervallo desiderato, anche se CoinGecko lo sceglie
    # in base a 'days' per market_chart. Per /ohlc, 'days' definisce la granularità.
    # Questo 'target_interval' è più un'indicazione per il post-processing se necessario.
    target_interval: str = "daily" # "daily", "hourly", "4hourly" etc.
) -> pd.DataFrame | None:
    logger.info(f"Chiamata API CG (cache) per {coin_id}, vs_curr: {vs_currency}, days: {days}, target_interval: {target_interval}")
    df_final = None
    
    # CoinGecko:
    # - /ohlc: se days <= 90, dà giornaliero. Se > 90, dà orario (o con granularità più fine).
    # - /market_chart: interval="daily" se days > 90. Se days <=90, dà granularità più fine (es. 5min, oraria).
    # Per semplicità, proviamo prima /ohlc se i giorni richiesti sono pochi (es. <= 7 per sperare in dati orari/sub-orari)
    # Altrimenti, per dati giornalieri o periodi più lunghi, /market_chart è più affidabile per la lunghezza dello storico.

    # Se vogliamo dati con granularità fine (es. oraria)
    if target_interval in ["1H", "4H"] or days <= 7 : # Arbitrario, per dati molto recenti e granulari
        use_ohlc_for_fine_granularity = days <= 90 # L'endpoint ohlc dà dati orari se days > 90, ma può essere limitato.
                                                # Per dati orari, meglio chiedere pochi giorni.
                                                # Se chiediamo 1 giorno, ohlc potrebbe dare dati ogni 30min/1h
        if use_ohlc_for_fine_granularity:
            actual_days_for_ohlc = days if days <=90 else 90 # Limita per evitare troppi dati orari da /ohlc
            logger.debug(f"Tentativo fetch CoinGecko /ohlc (target_interval={target_interval}, days={actual_days_for_ohlc}) per {coin_id}.")
            params_ohlc = {"vs_currency": vs_currency, "days": str(actual_days_for_ohlc)}
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
                    
                    # Volume da market_chart
                    # ... (come prima, ma considera che i giorni potrebbero essere pochi qui)
                    params_mc_vol = {"vs_currency": vs_currency, "days": str(actual_days_for_ohlc), "interval": "daily" if actual_days_for_ohlc > 1 else ""} # "" per auto-interval
                    resp_vol = requests.get(f"{CG_BASE_URL}/coins/{coin_id}/market_chart", params=params_mc_vol, timeout=20)
                    resp_vol.raise_for_status()
                    data_mc_vol = resp_vol.json()
                    if 'total_volumes' in data_mc_vol and data_mc_vol['total_volumes']:
                        df_v = pd.DataFrame(data_mc_vol['total_volumes'], columns=['Timestamp', 'Volume'])
                        df_v['Date'] = pd.to_datetime(df_v['Timestamp'], unit='ms')#.dt.normalize() # Non normalizzare per unire con intraday
                        df_v.set_index('Date', inplace=True)
                        # Per unire dati orari/sub-orari con volumi giornalieri, dovremmo fare un reindex/fillna
                        # Per ora, se stiamo prendendo dati intraday, il volume potrebbe essere sparso.
                        # Se i dati da /ohlc sono orari, e i volumi da market_chart sono giornalieri, l'unione non sarà perfetta.
                        # Per semplicità, se i dati sono < giornalieri, potremmo non avere volumi accurati per candela.
                        # O se /ohlc dà dati orari, e market_chart con interval="" dà volumi orari, allora va bene.
                        df_temp = df_temp.merge(df_v[['Volume']], left_index=True, right_index=True, how='left')
                        df_temp['Volume'].fillna(method='ffill', inplace=True) # Prova a propagare il volume se giornaliero
                        df_temp['Volume'].fillna(0, inplace=True)
                    else:
                        df_temp['Volume'] = 0.0
                    df_final = df_temp
                    logger.debug(f"Dati da CG /ohlc (gran. fine). Shape: {df_final.shape if df_final is not None else 'None'}")
            except Exception as e_ohlc_fine:
                logger.warning(f"CG /ohlc (gran. fine) fallito: {e_ohlc_fine}. Tentativo con /market_chart.")
    
    # Se si vogliono dati giornalieri, o periodo lungo, o il tentativo /ohlc sopra è fallito
    if df_final is None:
        logger.debug(f"Uso CG /market_chart per {coin_id} (days={days}, target_interval={target_interval}).")
        # Per market_chart, se days > 90, interval è implicitamente daily.
        # Se days <= 90, dà granularità più fine. Se vogliamo daily, dobbiamo specificare interval="daily".
        mc_interval = "daily" if target_interval == "daily" or days > 90 else "" # "" per auto-granularity
        params_market_chart = {"vs_currency": vs_currency, "days": str(days), "interval": mc_interval}
        url_market_chart = f"{CG_BASE_URL}/coins/{coin_id}/market_chart"
        try:
            response_mc = requests.get(url_market_chart, params=params_market_chart, timeout=30)
            response_mc.raise_for_status()
            data_market = response_mc.json()

            if 'prices' in data_market and data_market['prices']:
                df_prices = pd.DataFrame(data_market['prices'], columns=['Timestamp', 'Close'])
                df_prices['Date'] = pd.to_datetime(df_prices['Timestamp'], unit='ms') # Non normalizzare subito per intraday
                df_prices.set_index('Date', inplace=True)
                df_prices.drop(columns=['Timestamp'], inplace=True)

                df_final_mc = df_prices.copy()
                # Se market_chart non dà OHLC (cioè se abbiamo chiesto interval="daily" o se days è grande)
                # allora O, H, L = C. Se market_chart dà dati granulari (es. orari), questi sono solo "Close" orari.
                # Per un vero OHLC intraday, l'endpoint /ohlc è meglio se i giorni sono pochi.
                df_final_mc['Open'] = df_final_mc['Close'] 
                df_final_mc['High'] = df_final_mc['Close']
                df_final_mc['Low']  = df_final_mc['Close']

                if 'total_volumes' in data_market and data_market['total_volumes']:
                    # ... (logica volume come prima, assicurandosi che l'indice sia compatibile)
                    df_volumes = pd.DataFrame(data_market['total_volumes'], columns=['Timestamp', 'Volume'])
                    df_volumes['Date'] = pd.to_datetime(df_volumes['Timestamp'], unit='ms') # Non normalizzare
                    df_volumes.set_index('Date', inplace=True)
                    df_final_mc = df_final_mc.merge(df_volumes[['Volume']], left_index=True, right_index=True, how='left')
                    df_final_mc['Volume'].fillna(method='ffill', inplace=True)
                    df_final_mc['Volume'].fillna(0, inplace=True)
                else:
                    df_final_mc['Volume'] = 0.0
                
                df_final = df_final_mc[['Open', 'High', 'Low', 'Close', 'Volume']] 
                logger.debug(f"Dati da CG /market_chart. Shape: {df_final.shape if df_final is not None else 'None'}")
            else:
                logger.warning(f"CG /market_chart: Nessun dato 'prices' per {coin_id}.")
                return None 
        # ... (blocchi except come prima, usando logger) ...
        except requests.exceptions.HTTPError as e_mc: 
            if hasattr(e_mc, 'response') and e_mc.response is not None and e_mc.response.status_code == 429:
                logger.error(f"CG /market_chart: Rate limit (429) per {coin_id}. Dettagli: {e_mc.response.text[:200] if hasattr(e_mc.response, 'text') else 'N/A'}")
            else:
                logger.error(f"CG /market_chart: Errore HTTP per {coin_id}: {e_mc}. Response: {e_mc.response.text[:200] if hasattr(e_mc.response, 'text') else 'N/A'}")
            return None
        except Exception as e_mc:
            response_text_for_error_mc = response_mc.text[:200] if 'response_mc' in locals() and hasattr(response_mc, 'text') else 'N/A'
            logger.error(f"CG /market_chart: Errore caricamento per {coin_id}: {e_mc}. Risposta: {response_text_for_error_mc}", exc_info=True)
            return None


    if df_final is None or df_final.empty:
        logger.warning(f"CoinGecko: Nessun dato valido per {coin_id} dopo tutti i tentativi.")
        return None

    if 'Close' in df_final.columns:
        df_final['Adj_close'] = df_final['Close'] 
    
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj_close']
    for col in numeric_cols:
        if col in df_final.columns:
            df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
    
    df_final.dropna(subset=['Close'], inplace=True) 

    if df_final.empty:
        logger.warning(f"CoinGecko: Nessun dato valido per {coin_id} dopo processazione finale.")
        return None
    
    df_final = df_final.sort_index(ascending=True)
    logger.debug(f"Dati finali per {coin_id} caricati da CoinGecko. Shape: {df_final.shape}. Indice: {df_final.index.dtype}, Min: {df_final.index.min()}, Max: {df_final.index.max()}")
    return df_final


if __name__ == '__main__':
    # ... (blocco if __name__ == '__main__' come prima, usando logger) ...
    if 'logger' not in locals(): 
        import logging, sys # Aggiunto sys qui
        logger = logging.getLogger(__name__)
        if not logger.handlers: 
            handler = logging.StreamHandler(sys.stdout) 
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)

    logger.info("--- INIZIO TEST STANDALONE data_utils.py ---") 
    
    TEST_AV_API_KEY = None 
    if TEST_AV_API_KEY:
        logger.info("\n--- Test Alpha Vantage Intraday (60min) ---")
        df_stock_intra = get_stock_data(TEST_AV_API_KEY, "IBM", "TIME_SERIES_INTRADAY", "compact", av_interval="60min")
        if df_stock_intra is not None:
            logger.info("Dati IBM Intraday (Alpha Vantage):\n" + df_stock_intra.tail().to_string())
    else:
        logger.info("\nTest Alpha Vantage saltato: TEST_AV_API_KEY non impostata.")

    logger.info("\n--- Test CoinGecko ---")
    logger.info("Test per Bitcoin (giornaliero, 30 giorni):")
    df_crypto_btc_short = get_crypto_data(coin_id="bitcoin", vs_currency="usd", days=30, target_interval="daily")
    if df_crypto_btc_short is not None:
        logger.info("Dati Bitcoin (30 giorni):\n" + df_crypto_btc_short.tail().to_string())
    
    logger.info("\nTest per Ethereum (orario, ultimi 2 giorni):")
    df_crypto_eth_hourly = get_crypto_data(coin_id="ethereum", vs_currency="usd", days=2, target_interval="1H") 
    if df_crypto_eth_hourly is not None:
        logger.info("Dati Ethereum (Orari, 2 giorni):\n" + df_crypto_eth_hourly.tail().to_string())

    logger.info("\nTest per Cardano (>90 giorni, dovrebbe usare market_chart):")
    df_crypto_ada_long = get_crypto_data(coin_id="cardano", vs_currency="usd", days=120, target_interval="daily") 
    if df_crypto_ada_long is not None:
        logger.info("Dati Cardano (120 giorni):\n" + df_crypto_ada_long.tail().to_string())
    
    logger.info("\n--- FINE TEST STANDALONE data_utils.py ---")
