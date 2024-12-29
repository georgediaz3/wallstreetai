# src/fetch_data.py

"""
fetch_data.py

Functions to fetch and preprocess cryptocurrency data using CCXT.
This module is utilized by both preprocess.py and ui_app.py.
"""

import ccxt
import pandas as pd
from datetime import datetime
import streamlit as st  # For displaying error messages in Streamlit
import time  # To implement sleep for rate limiting

# Define the exchange ID consistently across modules
EXCHANGE_ID = "coinbaseadvanced"  # Changed to 'coinbasepro' as 'coinbaseadvanced' might not be recognized

def get_exchange(exchange_id=EXCHANGE_ID):
    """
    Initializes and returns the CCXT exchange instance.
    
    Parameters:
    - exchange_id (str): The CCXT exchange identifier.
    
    Returns:
    - ccxt.Exchange: An instance of the specified exchange.
    
    Raises:
    - ValueError: If the exchange is not supported by CCXT.
    - RuntimeError: For any other initialization errors.
    """
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class()
        exchange.load_markets()  # Load markets to respect rate limits and ensure symbols are loaded
        return exchange
    except AttributeError:
        raise ValueError(f"Exchange '{exchange_id}' is not supported by CCXT.")
    except Exception as e:
        raise RuntimeError(f"Error initializing exchange {exchange_id}: {e}")

def fetch_multiple_ohlcv(symbols=None, timeframe='1h', limit=100):
    """
    Fetches OHLCV data for multiple symbols and returns a combined DataFrame with a 'symbol' column.
    
    Parameters:
    - symbols (list): List of trading pairs (e.g., ['BTC/USD', 'ETH/USD']).
    - timeframe (str): Timeframe for OHLCV data (e.g., '1h').
    - limit (int): Number of data points to fetch per symbol.
    
    Returns:
    - pd.DataFrame: Combined OHLCV data for all symbols.
    """
    if symbols is None:
        symbols = ['BTC/USD']  # Default fallback

    try:
        exchange = get_exchange()
    except (ValueError, RuntimeError) as e:
        st.error(e)
        return pd.DataFrame()

    all_dfs = []
    for sym in symbols:
        try:
            st.write(f"Fetching OHLCV data for {sym}...")
            data = exchange.fetch_ohlcv(sym, timeframe, limit=limit)
            time.sleep(0.2)  # Sleep to respect rate limits
            columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df = pd.DataFrame(data, columns=columns)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = sym
            all_dfs.append(df)
            st.write(f"Fetched {len(df)} rows for {sym}.")
        except Exception as e:
            st.error(f"Error fetching OHLCV data for {sym}: {e}")
            st.write(f"Skipping {sym} due to error.")
            continue  # Skip to the next symbol

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.sort_values(['symbol', 'timestamp'], inplace=True)
        st.write("Successfully fetched and combined OHLCV data for all symbols.")
        return combined
    else:
        st.error("No OHLCV data fetched for any symbols.")
        return pd.DataFrame()

def fetch_realtime_quotes(symbols=None):
    """
    Fetches real-time ticker data (price, change, volume, etc.) for each symbol.
    Returns a DataFrame with columns: symbol, price, change_24h_%, volume_24h, timestamp
    
    Parameters:
    - symbols (list): List of trading pairs.
    
    Returns:
    - pd.DataFrame: Real-time ticker data for the symbols.
    """
    if symbols is None:
        symbols = ["BTC/USD"]

    try:
        exchange = get_exchange()
    except (ValueError, RuntimeError) as e:
        st.error(e)
        return pd.DataFrame()

    results = []
    for sym in symbols:
        try:
            st.write(f"Fetching ticker for {sym}...")
            ticker = exchange.fetch_ticker(sym)
            time.sleep(0.2)  # Sleep to respect rate limits

            # Extract relevant data
            price = ticker.get('last')
            change_pct = ticker.get('percentage')
            volume_24h = ticker.get('baseVolume')

            # Ensure all values are floats; if not, set defaults
            price = float(price) if price is not None else 0.0
            change_pct = float(change_pct) if change_pct is not None else 0.0
            volume_24h = float(volume_24h) if volume_24h is not None else 0.0

            results.append({
                'symbol': sym,
                'price': price,
                'change_24h_%': change_pct,
                'volume_24h': volume_24h,
                'timestamp': datetime.now()
            })
            st.write(f"Fetched ticker for {sym}: Price=${price}, Change={change_pct}%, Volume={volume_24h}")
        except Exception as e:
            st.error(f"Error fetching ticker for {sym}: {e}")
            st.write(f"Skipping {sym} due to error.")
            continue  # Skip to the next symbol

    return pd.DataFrame(results)

def fetch_ohlcv(symbol='BTC/USD', timeframe='1m', limit=30):
    try:
        exchange = get_exchange()
        data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        st.error(f"Error fetching OHLCV data for {symbol}: {e}")
        return pd.DataFrame()

def get_all_symbols(quote_currencies=["USD", "USDT"]):
    """
    Retrieves all trading symbols from the exchange and filters them based on specified quote currencies.
    
    Parameters:
    - quote_currencies (list): List of quote currencies to filter symbols (e.g., ['USD', 'USDT']).
    
    Returns:
    - list: Filtered list of trading symbols.
    """
    try:
        exchange = get_exchange()
    except (ValueError, RuntimeError) as e:
        st.error(e)
        return []

    try:
        all_symbols = list(exchange.markets.keys())
        filtered_symbols = [s for s in all_symbols if any(s.endswith(f"/{quote}") for quote in quote_currencies)]
        st.write(f"Found {len(filtered_symbols)} symbols with specified quote currencies.")
        return filtered_symbols
    except Exception as e:
        st.error(f"Error retrieving symbols: {e}")
        return []
def filter_crypto_pairs(all_symbols, base_currencies=["BTC", "ETH"]):
    """
    Filters symbols to include only those with specified base currencies.

    Parameters:
    - all_symbols (list): List of trading pairs (e.g., ['BTC/USD', 'ETH/USD']).
    - base_currencies (list): List of base currencies to filter (e.g., ['BTC', 'ETH']).

    Returns:
    - list: Filtered list of symbols.
    """
    return [symbol for symbol in all_symbols if symbol.split('/')[0] in base_currencies]










