# src/preprocess.py

"""
preprocess.py

Script to preprocess cryptocurrency data by fetching OHLCV data,
computing technical indicators, performing sentiment analysis, and saving
the processed data to a CSV file.
"""

import os
import logging
import pandas as pd
import ccxt
from datetime import datetime
import time
import pickle

# Optional: Sentiment Analysis Libraries
from textblob import TextBlob  # Install via `pip install textblob`
# Alternatively, for more advanced sentiment analysis, consider using transformers.

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocess.log"),
        logging.StreamHandler()
    ]
)

# Define Constants
EXCHANGE_ID = "coinbaseadvanced"  # You can change this to any supported exchange in ccxt
TIMEFRAME = '1h'  # Timeframe for OHLCV data
LIMIT = 1000  # Number of data points to fetch per symbol
DATA_DIR = "data"
MODELS_DIR = "models"
DATA_FILE = os.path.join(DATA_DIR, "multi_data_with_indicators_sentiment.csv")
MODEL_FILE = os.path.join(MODELS_DIR, "random_forest.pkl")

# Define Symbols to Fetch
SYMBOLS = ["BTC/USD", "ETH/USD", "DOGE/USD", "SOL/USD", "ADA/USD"]  # Add more as needed

def ensure_directories():
    """Ensure that data and models directories exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    logging.info(f"Ensured that directories '{DATA_DIR}' and '{MODELS_DIR}' exist.")

def get_exchange(exchange_id=EXCHANGE_ID):
    """
    Initialize and return a CCXT exchange instance.

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
        exchange.load_markets()
        logging.info(f"Initialized exchange: {exchange_id}")
        return exchange
    except AttributeError:
        raise ValueError(f"Exchange '{exchange_id}' is not supported by CCXT.")
    except Exception as e:
        raise RuntimeError(f"Error initializing exchange {exchange_id}: {e}")

def fetch_ohlcv_data(exchange, symbol, timeframe=TIMEFRAME, limit=LIMIT):
    """
    Fetch OHLCV data for a single symbol.

    Parameters:
    - exchange (ccxt.Exchange): The CCXT exchange instance.
    - symbol (str): Trading pair symbol (e.g., 'BTC/USD').
    - timeframe (str): Timeframe for OHLCV data.
    - limit (int): Number of data points to fetch.

    Returns:
    - pd.DataFrame: DataFrame containing OHLCV data.
    """
    try:
        logging.info(f"Fetching OHLCV data for {symbol}...")
        data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        time.sleep(0.2)  # Respect rate limits
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['symbol'] = symbol
        logging.info(f"Fetched {len(df)} rows for {symbol}.")
        return df
    except Exception as e:
        logging.error(f"Error fetching OHLCV data for {symbol}: {e}")
        return pd.DataFrame()

def fetch_multiple_ohlcv(symbols=SYMBOLS, timeframe=TIMEFRAME, limit=LIMIT):
    """
    Fetch OHLCV data for multiple symbols and combine them into a single DataFrame.

    Parameters:
    - symbols (list): List of trading pair symbols.
    - timeframe (str): Timeframe for OHLCV data.
    - limit (int): Number of data points to fetch per symbol.

    Returns:
    - pd.DataFrame: Combined DataFrame with OHLCV data for all symbols.
    """
    exchange = get_exchange()
    all_data = []
    for symbol in symbols:
        df = fetch_ohlcv_data(exchange, symbol, timeframe, limit)
        if not df.empty:
            all_data.append(df)
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.sort_values(['symbol', 'timestamp'], inplace=True)
        logging.info("Successfully fetched and combined OHLCV data for all symbols.")
        return combined_df
    else:
        logging.error("No OHLCV data fetched for any symbols.")
        return pd.DataFrame()

def compute_technical_indicators(df):
    """
    Compute technical indicators using the 'ta' library.

    Parameters:
    - df (pd.DataFrame): DataFrame containing OHLCV data.

    Returns:
    - pd.DataFrame: DataFrame with additional technical indicator columns.
    """
    try:
        import ta  # Ensure 'ta' is installed: pip install ta
    except ImportError:
        logging.error("The 'ta' library is not installed. Install it using 'pip install ta'")
        raise

    logging.info("Computing technical indicators...")
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    macd = ta.trend.MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    bollinger = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()
    logging.info("Technical indicators computed successfully.")
    return df

def perform_sentiment_analysis(df):
    """
    Perform sentiment analysis on related news headlines for each symbol.

    Parameters:
    - df (pd.DataFrame): DataFrame containing OHLCV data.

    Returns:
    - pd.DataFrame: DataFrame with an additional 'sentiment' column.
    
    Note:
    - This function uses a placeholder for sentiment scores.
    - To implement actual sentiment analysis, integrate with news APIs or social media data sources.
    """
    logging.info("Performing sentiment analysis (placeholder)...")
    # Placeholder: Assign random sentiment scores or implement actual sentiment fetching
    # For demonstration, we'll assign a neutral sentiment score
    df['sentiment'] = 0.0  # Replace with actual sentiment scores
    logging.info("Sentiment analysis completed (placeholder).")
    return df

def create_target_variable(df):
    """
    Create a target variable for model training based on RSI.

    Parameters:
    - df (pd.DataFrame): DataFrame with technical indicators.

    Returns:
    - pd.DataFrame: DataFrame with an additional 'buy_signal' column.
    """
    logging.info("Creating target variable 'buy_signal' based on RSI...")
    # Example Strategy: Buy signal if RSI < 30
    df['buy_signal'] = (df['rsi'] < 30).astype(int)
    logging.info("Target variable 'buy_signal' created successfully.")
    return df

def preprocess_data(symbols=SYMBOLS, timeframe=TIMEFRAME, limit=LIMIT):
    """
    Full preprocessing pipeline: fetch data, compute indicators, perform sentiment analysis,
    create target variable, and save to CSV.

    Parameters:
    - symbols (list): List of trading pair symbols.
    - timeframe (str): Timeframe for OHLCV data.
    - limit (int): Number of data points to fetch per symbol.
    """
    try:
        logging.info("Starting data preprocessing pipeline...")
        # Fetch Data
        df = fetch_multiple_ohlcv(symbols, timeframe, limit)
        if df.empty:
            logging.error("No data fetched. Exiting preprocessing pipeline.")
            return

        # Compute Technical Indicators
        df = compute_technical_indicators(df)

        # Perform Sentiment Analysis (Optional)
        df = perform_sentiment_analysis(df)

        # Create Target Variable
        df = create_target_variable(df)

        # Drop rows with any missing values
        initial_length = len(df)
        df.dropna(inplace=True)
        final_length = len(df)
        logging.info(f"Dropped {initial_length - final_length} rows due to NaN values.")

        # Save to CSV
        df.to_csv(DATA_FILE, index=False)
        logging.info(f"Preprocessed data saved to '{DATA_FILE}' successfully.")

    except Exception as e:
        logging.error(f"An error occurred during preprocessing: {e}")

def main():
    """Main function to execute preprocessing."""
    ensure_directories()
    preprocess_data()

    # Optional: Train and Save Model After Preprocessing
    # Uncomment the lines below if you want to train the model immediately after preprocessing
    # from train_model import train_model
    # train_model(DATA_FILE, MODEL_FILE)

if __name__ == "__main__":
    main()

