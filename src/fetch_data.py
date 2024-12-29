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
    """
    Fetches short-term OHLCV data for a single symbol.
    Useful for generating real-time charts in the Streamlit app.
    
    Parameters:
    - symbol (str): Trading pair (e.g., 'BTC/USD').
    - timeframe (str): Timeframe for OHLCV data (e.g., '1m').
    - limit (int): Number of data points to fetch.
    
    Returns:
    - pd.DataFrame: OHLCV data for the symbol.
    """
    try:
        exchange = get_exchange()
    except (ValueError, RuntimeError) as e:
        st.error(e)
        return pd.DataFrame()

    try:
        st.write(f"Fetching OHLCV data for {symbol} with timeframe {timeframe}...")
        data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        time.sleep(0.2)  # Sleep to respect rate limits
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        st.write(f"Fetched {len(df)} rows for {symbol}.")
        return df
    except Exception as e:
        st.error(f"Error fetching OHLCV for {symbol}: {e}")
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
def compute_technical_indicators(df):
    """
    Compute technical indicators using the 'ta' library.
    """
    try:
        import ta  # Ensure 'ta' is installed: pip install ta
    except ImportError:
        st.error("The 'ta' library is required to compute indicators. Install it using `pip install ta`.")
        return df

    # Add RSI
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()

    # Add MACD
    macd = ta.trend.MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    # Add Bollinger Bands
    bollinger = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()

    return df
    def analyze_and_trade(symbols, model):
    for symbol in symbols:
        try:
            ohlcv_df = fetch_ohlcv(symbol, timeframe='1m', limit=30)  # Fetch last 30 minutes of data
            if ohlcv_df.empty:
                continue

            # Compute indicators
            ohlcv_df = compute_technical_indicators(ohlcv_df)
            ohlcv_df.dropna(inplace=True)  # Drop rows with NaN values due to indicator calculation

            # Extract features for AI model
            latest_features = ohlcv_df.iloc[-1][['rsi', 'macd', 'macd_signal', 'macd_diff', 'bb_high', 'bb_low']].values.reshape(1, -1)
            prediction = model.predict(latest_features)[0]
            current_price = ohlcv_df.iloc[-1]['close']

            # Make trade decision
            if prediction == 1:  # BUY signal
                trade_amount = st.session_state['paper_balance'] * 0.1  # Trade 10% of balance
                place_paper_buy(symbol, trade_amount, current_price)
            elif st.session_state['holdings'].get(symbol, 0) > 0:  # SELL signal
                sell_quantity = st.session_state['holdings'][symbol]
                place_paper_sell(symbol, sell_quantity, current_price)
        except Exception as e:
            st.error(f\"Error processing {symbol}: {e}\")









