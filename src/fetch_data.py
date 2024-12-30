import ccxt
import pandas as pd
import ta
from datetime import datetime
import time

# Define the exchange ID
EXCHANGE_ID = "coinbaseadvanced"  # Updated to 'coinbasepro' for better CCXT support

def get_exchange(exchange_id=EXCHANGE_ID):
    """
    Initializes and returns the CCXT exchange instance.

    Parameters:
    - exchange_id (str): The CCXT exchange identifier.

    Returns:
    - ccxt.Exchange: An instance of the specified exchange.
    """
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class()
        exchange.load_markets()
        return exchange
    except AttributeError:
        raise ValueError(f"Exchange '{exchange_id}' is not supported by CCXT.")
    except Exception as e:
        raise RuntimeError(f"Error initializing exchange {exchange_id}: {e}")

def fetch_multiple_ohlcv(symbols=None, timeframe='1h', limit=1000):
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
        print(f"Error initializing exchange: {e}")
        return pd.DataFrame()

    all_dfs = []
    for sym in symbols:
        try:
            print(f"Fetching OHLCV data for {sym}...")
            data = exchange.fetch_ohlcv(sym, timeframe, limit=limit)
            time.sleep(0.2)  # Respect rate limits
            columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df = pd.DataFrame(data, columns=columns)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = sym  # Add symbol column

            # Add technical indicators
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            bollinger = ta.volatility.BollingerBands(df['close'], window=20)
            df['bb_high'] = bollinger.bollinger_hband()
            df['bb_low'] = bollinger.bollinger_lband()

            all_dfs.append(df)
            print(f"Fetched {len(df)} rows for {sym}.")
        except Exception as e:
            print(f"Error fetching OHLCV data for {sym}: {e}")
            continue

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.sort_values(['symbol', 'timestamp'], inplace=True)
        print("Successfully fetched and combined OHLCV data for all symbols.")
        return combined
    else:
        print("No OHLCV data fetched for any symbols.")
        return pd.DataFrame()

def save_data(df, filename='data/historical_data_with_indicators.csv'):
    """
    Saves the fetched data to a CSV file in a format compatible with backtest.

    Parameters:
    - df (pd.DataFrame): The DataFrame to save.
    - filename (str): Path to save the CSV file.
    """
    if not df.empty:
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'rsi', 'macd', 'macd_signal', 'macd_diff', 'bb_high', 'bb_low']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
    else:
        print("No data to save.")

def main():
    """
    Main function to fetch and save OHLCV data.
    """
    symbols = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'DOGE/USD', 'ADA/USD']
    df = fetch_multiple_ohlcv(symbols=symbols, timeframe='1h', limit=2000)  # Increased limit for more data
    save_data(df)

if __name__ == "__main__":
    main()











