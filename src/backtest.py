import pandas as pd
import pickle

def backtest(
    data_path='data/historical_data_with_indicators.csv', 
    model_path='models/random_forest.pkl'
):
    # Load data
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Ensure timestamp is in datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Filter for the last week
    start_date = pd.Timestamp.now() - pd.Timedelta(days=7)
    df = df[df['timestamp'] >= start_date]

    if df.empty:
        print("No data available for the last week.")
        return

    # Add a placeholder for missing features
    if 'sentiment' not in df.columns:
        print("Warning: 'sentiment' feature missing. Adding a placeholder.")
        df['sentiment'] = 0.0  # Neutral sentiment placeholder

    if 'symbol' not in df.columns:
        print("Warning: 'symbol' column missing. Adding a placeholder.")
        df['symbol'] = 'Unknown'  # Add a placeholder value

    # Recreate 'target'
    df['future_close'] = df['close'].shift(-1)
    df.dropna(inplace=True)
    df['target'] = (df['future_close'] > df['close']).astype(int)

    # Define features
    features = ['rsi', 'macd', 'macd_signal', 'macd_diff', 'bb_high', 'bb_low', 'sentiment']
    if not all(feature in df.columns for feature in features):
        print("Error: Some required features are missing in the data.")
        return

    X = df[features]

    # Load the trained model
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Make predictions
    df['prediction'] = model.predict(X)

    # Calculate % change for the next candle
    df['next_close'] = df['close'].shift(-1)
    df.dropna(inplace=True)
    df['pct_change'] = (df['next_close'] - df['close']) / df['close']

    # Calculate strategy return
    df['strategy_return'] = df.apply(
        lambda row: row['pct_change'] if row['prediction'] == 1 else 0, 
        axis=1
    )

    # Calculate total return
    total_return = (df['strategy_return'] + 1).prod() - 1
    print(f"Total Hypothetical Return: {total_return * 100:.2f}%")

    # Identify trades
    df['trade_action'] = df.apply(
        lambda row: 'BUY' if row['prediction'] == 1 else 'HOLD',
        axis=1
    )
    trades = df[df['trade_action'] == 'BUY'][['timestamp', 'symbol', 'close', 'trade_action']]
    trades.to_csv('week_trades.csv', index=False)

    # Output trades
    print("Trades made over the last week:")
    print(trades)

if __name__ == "__main__":
    backtest()
