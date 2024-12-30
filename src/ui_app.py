import ccxt
import pandas as pd
import ta
import streamlit as st
from datetime import datetime
import time
import pickle
from sklearn.ensemble import RandomForestClassifier

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
        st.error(f"Error initializing exchange: {e}")
        return pd.DataFrame()

    all_dfs = []
    for sym in symbols:
        try:
            st.write(f"Fetching OHLCV data for {sym}...")
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
            st.write(f"Fetched {len(df)} rows for {sym}.")
        except Exception as e:
            st.error(f"Error fetching OHLCV data for {sym}: {e}")
            continue

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.sort_values(['symbol', 'timestamp'], inplace=True)
        st.write("Successfully fetched and combined OHLCV data for all symbols.")
        return combined
    else:
        st.error("No OHLCV data fetched for any symbols.")
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
        st.write(f"Data saved to {filename}")
    else:
        st.error("No data to save.")

def train_model(data_path='data/historical_data_with_indicators.csv', model_path='models/random_forest.pkl'):
    """
    Trains a Random Forest model on the given dataset.

    Parameters:
    - data_path (str): Path to the input data file.
    - model_path (str): Path to save the trained model.
    """
    try:
        df = pd.read_csv(data_path)
        df.dropna(inplace=True)

        # Define features and target
        features = ['rsi', 'macd', 'macd_signal', 'macd_diff', 'bb_high', 'bb_low']
        X = df[features]
        y = (df['close'].shift(-1) > df['close']).astype(int)  # Binary target: 1 if next close > current close

        # Train Random Forest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X[:-1], y[:-1])  # Exclude the last row for training

        # Save the model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        st.write("Model trained and saved successfully!")
    except Exception as e:
        st.error(f"Error during model training: {e}")

def backtest(data_path='data/historical_data_with_indicators.csv', model_path='models/random_forest.pkl', start_date=None, end_date=None):
    """
    Backtests the trained model on historical data.

    Parameters:
    - data_path (str): Path to the historical data file.
    - model_path (str): Path to the trained model.
    - start_date (str): Start date for the backtest in 'YYYY-MM-DD' format.
    - end_date (str): End date for the backtest in 'YYYY-MM-DD' format.
    """
    try:
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Convert Streamlit date inputs to datetime64
        if start_date:
            start_date = pd.Timestamp(start_date)
            df = df[df['timestamp'] >= start_date]
        if end_date:
            end_date = pd.Timestamp(end_date)
            df = df[df['timestamp'] <= end_date]

        if df.empty:
            st.error("No data available for the specified date range.")
            return

        # Load the model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Define features and make predictions
        features = ['rsi', 'macd', 'macd_signal', 'macd_diff', 'bb_high', 'bb_low']
        df['prediction'] = model.predict(df[features])

        # Calculate returns
        df['next_close'] = df['close'].shift(-1)
        df.dropna(inplace=True)
        df['strategy_return'] = df.apply(
            lambda row: (row['next_close'] - row['close']) / row['close'] if row['prediction'] == 1 else 0,
            axis=1
        )

        total_return = (df['strategy_return'] + 1).prod() - 1
        st.write(f"Total Hypothetical Return: {total_return * 100:.2f}%")

        # Show trades
        trades = df[df['prediction'] == 1][['timestamp', 'symbol', 'close']]
        st.write("Trades Executed:")
        st.dataframe(trades)

    except Exception as e:
        st.error(f"Error during backtesting: {e}")

import os

def paper_trade(symbols, timeframe='1h', interval=3600, model_path='models/random_forest.pkl', trade_log='data/trade_log.csv'):
    """
    Executes live paper trading by fetching data, making predictions, and simulating trades.

    Parameters:
    - symbols (list): List of symbols to trade (e.g., ['BTC/USD', 'ETH/USD']).
    - timeframe (str): Timeframe for fetching data (e.g., '1h').
    - interval (int): Time interval (in seconds) between fetches.
    - model_path (str): Path to the trained model file.
    - trade_log (str): Path to save trade logs.
    """
    try:
        # Load the model
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model file not found. Train the model first.")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Initialize trading log
        if not os.path.exists(trade_log):
            with open(trade_log, 'w') as f:
                f.write("timestamp,symbol,action,price\n")

        st.write("Starting live paper trading...")

        while True:
            trades = []
            for symbol in symbols:
                # Fetch the latest data
                df = fetch_multiple_ohlcv(symbols=[symbol], timeframe=timeframe, limit=20)
                if df.empty:
                    st.warning(f"No data fetched for {symbol}. Skipping.")
                    continue

                # Prepare the latest row for prediction
                features = ['rsi', 'macd', 'macd_signal', 'macd_diff', 'bb_high', 'bb_low']
                latest_row = df[features].iloc[-1:].dropna()
                if latest_row.empty:
                    st.warning(f"No valid features for {symbol}. Skipping.")
                    continue

                # Make prediction
                prediction = model.predict(latest_row)[0]
                action = "BUY" if prediction == 1 else "HOLD"

                # Log the trade
                price = df['close'].iloc[-1]
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                trades.append({"timestamp": timestamp, "symbol": symbol, "action": action, "price": price})
                st.write(f"{timestamp} - {symbol}: {action} at {price}")

            # Save trades to log
            if trades:
                trade_df = pd.DataFrame(trades)
                trade_df.to_csv(trade_log, mode='a', header=False, index=False)
                st.write("Trades logged successfully.")

            # Wait for the next interval
            st.write(f"Sleeping for {interval} seconds...")
            time.sleep(interval)

    except Exception as e:
        st.error(f"Error during paper trading: {e}")


def main():
    """
    Main Streamlit UI for fetching data, training models, backtesting, and displaying results.
    """
    st.title("Crypto Data Fetching, Model Training, and Backtesting")

    # User inputs for fetching data
    symbols = st.text_input("Enter symbols (comma-separated):", value="BTC/USD,ETH/USD,SOL/USD,DOGE/USD,ADA/USD")
    timeframe = st.selectbox("Select timeframe:", options=['1m', '5m', '1h', '4h', '1d'], index=2)
    limit = st.number_input("Enter data limit per symbol:", value=2000, min_value=100, max_value=5000, step=100)

    # Fetch data button
    if st.button("Fetch Data"):
        symbol_list = [s.strip() for s in symbols.split(',')]
        df = fetch_multiple_ohlcv(symbols=symbol_list, timeframe=timeframe, limit=limit)
        if not df.empty:
            save_data(df)
        else:
            st.error("No data was fetched. Please check your inputs or try again.")

    # Train model button
    if st.button("Train Model"):
        train_model()

    # Backtest inputs
    start_date = st.date_input("Backtest Start Date:", value=None)
    end_date = st.date_input("Backtest End Date:", value=None)

    # Backtest button
    if st.button("Run Backtest"):
        backtest(start_date=start_date, end_date=end_date)

    if st.button("Start Paper Trading"):
    symbols = [s.strip() for s in symbols.split(',')]
    paper_trade(symbols=symbols, timeframe=timeframe)


if __name__ == "__main__":
    main()













